from pathlib import Path
from typing import Any

import torch
from pynvml import *
from transformers import (DataCollatorForLanguageModeling, PreTrainedModel,
                          PreTrainedTokenizerBase, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)


class ManualAdapterSaveCallback(TrainerCallback):
    """A TrainerCallback to manually save LoRA adapters at the end of each epoch."""
    def __init__(self, adapter_base_save_dir: Path):
        super().__init__()
        self.adapter_base_save_dir = Path(adapter_base_save_dir)

    def on_epoch_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        model: PreTrainedModel | None = None,
        **kwargs
    ):
        """Saves the PEFT model to a subdirectory named after the current epoch."""
        if model is not None and hasattr(model, 'save_pretrained'):
            epoch = int(state.epoch)
            epoch_save_path = self.adapter_base_save_dir / f"epoch_{epoch}"
            epoch_save_path.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(epoch_save_path, save_embedding_layers=False)



class GpuMemoryCallbackIntegrated(TrainerCallback):
    """A TrainerCallback to monitor and log GPU memory usage during training."""
    def __init__(self, log_interval_steps=50):
        super().__init__()
        self.peak_memory_overall_mb = 0
        self.peak_memory_epoch_mb = 0
        self.log_interval_steps = log_interval_steps
        self._nvml_initialized = False

    def _init_nvml(self):
        if not self._nvml_initialized:
            try:
                nvmlInit()
                self._nvml_initialized = True
            except NVMLError as e:
                print(f"NVML Init Error: {e}. GPU stats won't be available.")

    def _shutdown_nvml(self):
        if self._nvml_initialized:
            try:
                nvmlShutdown()
                self._nvml_initialized = False
            except NVMLError:
                pass # Ignore if already shut down

    def _get_gpu_memory_used_mb(self):
        if not self._nvml_initialized:
            return 0 
        
        used_mb = 0
        try:
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            used_mb = info.used // (1024**2)
            
            if used_mb > self.peak_memory_overall_mb:
                self.peak_memory_overall_mb = used_mb
            if used_mb > self.peak_memory_epoch_mb:
                self.peak_memory_epoch_mb = used_mb
        except NVMLError as e:
            # print(f"NVML Error getting GPU info: {e}") # Becomes verbose
            pass 
        return used_mb

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._init_nvml()
        self.peak_memory_overall_mb = 0
        current_mem = self._get_gpu_memory_used_mb()
        print(f"GPU Memory at TRAIN BEGIN: {current_mem} MB")

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.peak_memory_epoch_mb = 0
        current_mem = self._get_gpu_memory_used_mb()
        print(f"GPU Memory at EPOCH {state.epoch:.0f} BEGIN: {current_mem} MB")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        current_mem = self._get_gpu_memory_used_mb() 

        if state.global_step > 0 and state.global_step % self.log_interval_steps == 0:
            print(f"  Step {state.global_step}: GPU Mem: {current_mem} MB (Epoch Peak: {self.peak_memory_epoch_mb} MB, Overall Peak: {self.peak_memory_overall_mb} MB)")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        current_mem = self._get_gpu_memory_used_mb()
        print(f"GPU Memory at EPOCH {state.epoch:.0f} END: {current_mem} MB. Final Peak for this epoch: {self.peak_memory_epoch_mb} MB.")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        current_mem = self._get_gpu_memory_used_mb()
        print(f"GPU Memory at TRAIN END: {current_mem} MB")
        print(f"OVERALL PEAK GPU Memory during training: {self.peak_memory_overall_mb} MB")
        self._shutdown_nvml()



class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    A custom data collator for causal language modeling that correctly handles
    pre-computed labels with padding.
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase, 
                 mlm: bool = False,
                 pad_to_multiple_of: int | None = None,):
        

        super().__init__(tokenizer=tokenizer, mlm=mlm, pad_to_multiple_of=pad_to_multiple_of)
        

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        original_labels = [example.pop("labels") for example in examples if "labels" in example]
        if len(original_labels) != len(examples):
            raise ValueError("All examples must have a 'labels' key to be popped.")


        batch = self.tokenizer.pad(
            examples,
            padding='longest',
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]

        pad_token_id = self.tokenizer.pad_token_id
        processed_and_padded_labels_list = []
        
        for label_seq in original_labels:
            # Replace pad_token_id with -100
            current_processed_label_seq = []
            if pad_token_id is not None:
                for token_id in label_seq:
                    if token_id == pad_token_id:
                        current_processed_label_seq.append(-100)
                    else:
                        current_processed_label_seq.append(token_id)
            else:
                # If no pad_token_id, or it's None, just use the label_seq as is
                current_processed_label_seq = list(label_seq)

            # Pad or truncate the label sequence
            diff = sequence_length - len(current_processed_label_seq)
            if diff > 0: # Label sequence is shorter
                processed_and_padded_labels_list.append(current_processed_label_seq + [-100] * diff)
            elif diff < 0: # Label sequence is longer
                processed_and_padded_labels_list.append(current_processed_label_seq[:sequence_length])
            else: # Lengths match
                processed_and_padded_labels_list.append(current_processed_label_seq)
        
        batch["labels"] = torch.tensor(processed_and_padded_labels_list, dtype=torch.long)
        
        if self.mlm:
             raise ValueError("MLM is True, but this custom collator is designed for Causal LM with pre-masked labels.")

        return batch