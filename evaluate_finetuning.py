import gc
import json
import logging
import shutil
import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from data.utils import TextDataset, create_collate_fn
from src.logging_config import setup_logging
from src.output_similarity import (AVAILABLE_METRICS, DERIVED_METRICS_SOURCES,
                                   HIGHER_IS_BETTER, compute_similarity_scores)
from src.utils import (find_best_candidate_by_rank, get_gpu_utilization,
                       set_seed)

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('filelock').setLevel(logging.WARNING)
logging.getLogger('accelerate').setLevel(logging.WARNING)
logging.getLogger('bitsandbytes').setLevel(logging.WARNING)
console = Console()

def get_args() -> Namespace:
    """Parses and validates command-line arguments for the finetuning evaluation script."""
    parser = ArgumentParser(
        description="Script for evaluating finetuned adapters.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where finetune.py saved its results."
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=list(HIGHER_IS_BETTER.keys()),
        default=["sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity"],
        help="List of metrics to use for evaluation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for generating model outputs during evaluation."
    )
    # Generation parameters for evaluation
    parser.add_argument("--max_new_tokens", type=int, default=125, help="Max new tokens for generation during evaluation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling.")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k for sampling.")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of sequences to return per prompt.")

    args = parser.parse_args()
    
    # Validate metrics
    valid_metrics = list(AVAILABLE_METRICS.keys()) + list(DERIVED_METRICS_SOURCES.keys())
    for metric in args.metrics:
        if metric not in valid_metrics:
            parser.error(f"Invalid metric: {metric}. Choices are: {valid_metrics}")
    return args

def _load_and_prepare_peft_model(
    model_name: str,
    adapter_path: Path,
    bnb_config: BitsAndBytesConfig | None,
    tokenizer: AutoTokenizer,
    compute_dtype: torch.dtype,
) -> AutoModelForCausalLM:
    """
    Loads a base model, merges a PEFT adapter, and reloads it for efficient inference.

    This function performs a multi-step process required for evaluating quantized LoRA models:
    1. Loads the full-precision base model.
    2. Merges the specified LoRA adapter into it.
    3. Saves the merged model to a temporary directory.
    4. Deletes the full model to free GPU memory.
    5. Loads the merged model from the temporary directory with quantization enabled.
    6. Cleans up the temporary directory.

    Args:
        model_name (str): The name or path of the base Hugging Face model.
        adapter_path (Path): Path to the directory containing the LoRA adapter.
        bnb_config (BitsAndBytesConfig | None): Configuration for BitsAndBytes quantization.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        compute_dtype (torch.dtype): The data type for model computations (e.g., bfloat16).

    Returns:
        (AutoModelForCausalLM): The final, inference-ready model with the adapter merged.
    """
    merged_model_dir = adapter_path / "peft_model_merged"

    try:
        # Step 1 & 2: Load base model and merge adapter
        logger.debug(f"Loading base model {model_name} to merge with adapter at {adapter_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        
        merged_model = PeftModel.from_pretrained(base_model, str(adapter_path)).merge_and_unload()
        logger.debug("Adapter merged successfully.")

        # Step 3: Save the merged model
        merged_model.save_pretrained(merged_model_dir)
        logger.debug(f"Temporarily saved merged model to {merged_model_dir}")

        # Step 4: Clean up memory
        del base_model, merged_model
        gc.collect()
        torch.cuda.empty_cache()

        # Step 5: Load the merged model for inference (with quantization)
        logger.debug("Loading merged model for evaluation...")
        inference_model = AutoModelForCausalLM.from_pretrained(
            merged_model_dir,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        inference_model.eval()
        return inference_model

    finally:
        # Step 6: Ensure cleanup happens even if errors occur
        if merged_model_dir.exists():
            shutil.rmtree(merged_model_dir)
            logger.debug(f"Cleaned up temporary directory: {merged_model_dir}")


def generate_model_responses_manual(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    generation_args: dict[str, Any],
) -> list[list[str]]:
    """
    Generates text responses for a given dataset using a model.

    Args:
        model (AutoModelForCausalLM): The model to use for generation.
        tokenizer (AutoTokenizer): The tokenizer for decoding the output.
        dataloader (DataLoader): DataLoader providing tokenized input batches.
        generation_args (dict[str, Any]): A dictionary of arguments for the `model.generate()` method.

    Returns:
        (list[list[str]]): A list of generated responses. Each inner list contains
                           `num_return_sequences` strings for a single input prompt.
    """
    all_responses = []
    gpu_memory_used = []
    device = model.device
    for batch_data in tqdm(dataloader, desc="Generating responses"):
        input_ids_batch = batch_data['input_ids']
        attention_mask_batch = batch_data['attention_mask']

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids_batch.to(device, non_blocking=True),
                attention_mask=attention_mask_batch.to(device, non_blocking=True),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                remove_invalid_values=True,
                **generation_args
            ).cpu()

        # Reshape the output to group sequences by original prompt
        num_return_sequences = generation_args.get('num_return_sequences', 1)
        current_batch_size = output_sequences.shape[0] // num_return_sequences
        model_output = output_sequences.view(current_batch_size, num_return_sequences, -1)

        # The length of the input prompt to slice off from the generated sequence
        response_start_index = input_ids_batch.shape[1]

        decoded_outputs = []
        for output_list in model_output:
            output_list = output_list[:, response_start_index:]
            decoded_outputs.append(tokenizer.batch_decode(output_list, skip_special_tokens=True))
        
        all_responses.extend(decoded_outputs)

        if torch.cuda.is_available():
            gpu_memory_used.append(get_gpu_utilization())
        
    if gpu_memory_used:
        logger.info(f'Max GPU memory occupied during output generation: {np.max(gpu_memory_used)//1024**2} MB')
    else:
        logger.debug('GPU memory usage not tracked for output generation (no CUDA).')
    return all_responses

def main(
    results_dir: str,
    metrics: list[str],
    eval_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_return_sequences: int,
) -> None:
    """
    Evaluates fine-tuned LoRA adapters by comparing their output to a conventional baseline.

    This script performs the following steps:
    1. Loads configuration and test data from a `finetune.py` results directory.
    2. Generates reference outputs using the conventional system prompt.
    3. Iterates through each saved LoRA adapter:
       - Merges the adapter with the base model.
       - Generates outputs on the test set.
       - Computes similarity scores against the reference outputs.
    4. Identifies the best-performing adapter based on ranked metrics.
    5. Saves the best adapter, its scores, and all evaluation artifacts.

    Args:
        results_dir (str): Path to the directory where finetune.py saved its results.
        metrics (list[str]): List of metrics to use for evaluation.
        eval_batch_size (int): Batch size for model generation.
        max_new_tokens (int): Max new tokens for generation.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling top-p.
        top_k (int): Sampling top-k.
        num_return_sequences (int): Number of sequences to generate per prompt.
    """
    #Step 1
    logger = logging.getLogger(__name__)
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    logger.info(f"Starting evaluation for results in: {results_dir}")
    params_file = results_dir / "params.json"
    if not params_file.exists():
        logger.error(f"params.json not found in {results_dir}")
        sys.exit(1)
    with open(params_file, "r") as f:
        params = json.load(f)
    logger.info(f"Loaded finetuning parameters: {json.dumps(params, indent=2)}")

    set_seed(params.get("seed", 42))

    quantization_mode = None
    if params.get("quantize_4bit", False):
        quantization_mode = "4bit"
    elif params.get("quantize_8bit", False):
        quantization_mode = "8bit"
    
    lora_adapters_path = results_dir / "lora_adapters"
    if not lora_adapters_path.is_dir():
        logger.error(f"lora_adapters directory not found in {results_dir}")
        sys.exit(1)

    # Have to load model manually otherwise we can not delete it later
    model_name = params["model_name"]
    logger.debug(f"Loading model: {model_name} with quantization: {quantization_mode or 'None'}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        padding_side="left",
    )
    new_pad_token = "<|pad|>"
    tokenizer.add_special_tokens({"pad_token": new_pad_token})

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.debug("CUDA capability >= 8.0, using bfloat16 for compute.")
            compute_dtype = torch.bfloat16
        else:
            logger.debug("CUDA capability < 8.0 or CUDA not available, using float16 for compute.")
            compute_dtype = torch.float16
    
    if quantization_mode == "4bit":
        logger.debug("Configuring for 4-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quantization_mode == "8bit":
        logger.debug("Configuring for 8-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        logger.debug("No quantization requested or unsupported mode.")
        bnb_config = None

    model_kwargs = {
        "device_map": "cpu",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype
    

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    model.to("cuda")
    model.eval()
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Loaded model: {model_name}")

    test_data_file = results_dir / "prepared_data" / "test_data.json"
    if not test_data_file.exists():
        logger.error(f"test_data.json not found in {results_dir / 'prepared_data'}")
        sys.exit(1)
    with open(test_data_file, "r") as f:
        test_user_prompts = json.load(f)

    logger.info(f"Loaded test data with {len(test_user_prompts)} prompts.")

    pad_token_string = tokenizer.pad_token
    system_prompt = params.get("system_prompt", f"{pad_token_string}{pad_token_string}")

    generation_config_eval = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
    }

    #Step 2
    logger.info("Generating reference outputs on test data using the conventional system prompt...")
    test_dataset = TextDataset(test_user_prompts)

    conventional_collate_fn = create_collate_fn(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )

    test_dataloader_conventional = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=conventional_collate_fn,
        shuffle=False
    )

    conventional_sys_output = generate_model_responses_manual(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_dataloader_conventional,
        generation_args=generation_config_eval
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    test_dataset = TextDataset(test_user_prompts)

    finetuning_collate_fn = create_collate_fn(
        tokenizer=tokenizer,
        system_prompt="",
    )

    test_dataloader_finetuning = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=finetuning_collate_fn,
        shuffle=False
    )

    #Step 3
    scores_list = []
    logger.debug(f"Finding best finetuned adapter...")

    for adapter_idx, adapter_subdir in enumerate(sorted(lora_adapters_path.iterdir())):
        if adapter_subdir.is_dir():
            adapter_name = adapter_subdir.name

            logger.info(f"Processing adapter: {adapter_name}")

            peft_model = _load_and_prepare_peft_model(model_name, adapter_subdir, bnb_config, tokenizer, compute_dtype)

            finetuning_output = generate_model_responses_manual(
                model=peft_model,
                tokenizer=tokenizer,
                dataloader=test_dataloader_finetuning,
                generation_args=generation_config_eval
            )

            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

            logger.debug(f"Calculating similarity scores...")
            scores = compute_similarity_scores(
                predictions=finetuning_output,
                references=conventional_sys_output,
                metric_list=metrics
            )
            logger.info(f"Similarity scores: {scores}")
            scores_list.append(scores)


    # Step 4
    best_idx, best_scores_dict = find_best_candidate_by_rank(
        candidate_scores=scores_list,
        metric_list=metrics,
        higher_is_better_map=HIGHER_IS_BETTER
    )
    best_adapter = sorted(lora_adapters_path.iterdir())[best_idx].name
    best_adapter_path = sorted(lora_adapters_path.iterdir())[best_idx]
    params['best_candidate_idx'] = best_idx
    logger.info(f"Best adapter: {best_adapter} with scores: {best_scores_dict}")

    logger.info("Regenerating output for the best adapter...")
    best_adapter_model = _load_and_prepare_peft_model(
        model_name,
        best_adapter_path,
        bnb_config,
        tokenizer,
        compute_dtype
    )

    best_finetuning_output = generate_model_responses_manual(
        model=best_adapter_model,
        tokenizer=tokenizer,
        dataloader=test_dataloader_finetuning,
        generation_args=generation_config_eval
    )

    del best_adapter_model
    gc.collect()
    torch.cuda.empty_cache()

    conventional_output_dict = {
        'output': conventional_sys_output,
        'input': test_user_prompts,
        'generation_config': generation_config_eval,
        'seed': params['seed']
    }

    best_finetuning_output_dict = {
        'output': best_finetuning_output,
        'input': test_user_prompts,
        'generation_config': generation_config_eval,
        'seed': params['seed']
    }

    # Step 5
    logger.debug(f"Saving best adapter...")
    shutil.copytree(str(best_adapter_path), str(results_dir / "best_adapter"))

    with open(results_dir / "best_adapter_scores.json", "w") as f:
        json.dump(best_scores_dict, f, indent=4)

    with open(results_dir / "best_adapter_output.json", "w") as f:
        json.dump(best_finetuning_output_dict, f, indent=4)

    with open(results_dir / "conventional_output.json", "w") as f:
        json.dump(conventional_output_dict, f, indent=4)

    with open(results_dir / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(results_dir / "all_scores.json", "w") as f:
        json.dump(scores_list, f, indent=4)

    with open(results_dir / "generation_config.json", "w") as f:
        json.dump(generation_config_eval, f, indent=4) 
    

    



if __name__ == "__main__":
    setup_logging('evaluate_finetuning.log', 'INFO') # Change to 'DEBUG' for more verbose logging
    logger = logging.getLogger(__name__)

    logger.debug("Parsing command line arguments...")
    try:
        args = get_args()
        logger.info(f"Command line arguments received: {json.dumps(vars(args), indent=2)}")
        main(**vars(args))
    except SystemExit:
        logger.warning("Exiting due to argument parsing issue (e.g., --help or invalid arguments).")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An critical error occurred: {e}")
        sys.exit(1)
    finally:
        logger.info("Done.")