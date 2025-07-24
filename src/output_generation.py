import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import Model
from src.prompt_utils import *
from src.utils import get_gpu_utilization

logger = logging.getLogger(__name__)


def precompute_model_outputs(
    model_wrapper: Model,
    dataloader: DataLoader,
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Precomputes model output probabilities and generated token IDs for a given dataset.

    This function iterates through a dataloader, generates outputs using greedy search
    (to get deterministic logits and IDs), and stacks the results into tensors.
    These precomputed outputs serve as the "ground truth" for the obfuscation optimization process.

    Args:
        model_wrapper (Model): The Model wrapper instance.
        dataloader (DataLoader): DataLoader yielding batches of tokenized input.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        (tuple[torch.Tensor, torch.Tensor, int]): A tuple containing:
            - A tensor of log probabilities for each generated token.
            - A tensor of the generated token IDs.
            - The maximum number of tokens generated across all samples.
    """
    probs_list = []
    ids_list = []
    device = model_wrapper.device
    gpu_memory_used = []
    max_generated_length = 0
    total_samples_processed = 0
    for data_batch in tqdm(dataloader, desc="Precomputing model outputs"):
        input_ids_batch = data_batch['input_ids']
        attention_mask_batch = data_batch['attention_mask']
        current_batch_size = input_ids_batch.shape[0]

        outputs = model_wrapper.generate_logits(
            input_tensor=input_ids_batch.to(device, non_blocking=True),
            attention_mask=attention_mask_batch.to(device, non_blocking=True),
            token_count=max_new_tokens,
            embedded=False
        )

        batch_logits = torch.stack(outputs.scores, dim=0).detach().cpu()
        gen_len_this_batch = batch_logits.shape[0]
        if gen_len_this_batch > max_generated_length:
            max_generated_length = gen_len_this_batch
        batch_probs = torch.nn.functional.log_softmax(batch_logits, dim=-1)
        batch_generated_ids = outputs.sequences[:, input_ids_batch.shape[1]:].detach().cpu()
        probs_list.append(batch_probs)
        ids_list.append(batch_generated_ids)

        if torch.cuda.is_available():
            gpu_memory_used.append(get_gpu_utilization())
    
        total_samples_processed += current_batch_size

    if gpu_memory_used:
        logger.info(f'Max GPU memory occupied during precomputation: {np.max(gpu_memory_used)//1024**2} MB')
    else:
        logger.info('GPU memory usage not tracked (either no CUDA or no batches processed).')

    final_probs_tensor = torch.zeros(
        (max_generated_length, total_samples_processed, model_wrapper.vocab_size),
        dtype=probs_list[0].dtype
    )
    final_ids_tensor = torch.full(
        (max_generated_length, total_samples_processed),
        model_wrapper.tokenizer.pad_token_id,
        dtype=ids_list[0].dtype
    )

    # Reshape the probs and ids tensors
    current_sample_idx = 0
    for i in range(len(probs_list)):
        batch_p = probs_list[i]
        batch_i = ids_list[i]
        gen_len_b = batch_p.shape[0]
        batch_sz_b = batch_p.shape[1]

        final_probs_tensor[:gen_len_b, current_sample_idx : current_sample_idx + batch_sz_b, :] = batch_p
        final_ids_tensor[:gen_len_b, current_sample_idx : current_sample_idx + batch_sz_b] = batch_i.transpose(0, 1)
        
        current_sample_idx += batch_sz_b

    return final_probs_tensor, final_ids_tensor, max_generated_length


def precompute_model_outputs_replace(
    model_wrapper: Model,
    dataloader: DataLoader,
    max_new_tokens: int,
    sys_prompt_obf: torch.Tensor,
    original_sys_prompt_ids: torch.Tensor,
    is_soft_prompt_method: bool,
    obf_sys_prompt_len: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Precomputes model outputs after replacing the system prompt with an obfuscated version.

    Similar to `precompute_model_outputs`, but first modifies the input batch by
    swapping the original system prompt with the provided obfuscated prompt (either
    token IDs or an embedding).

    Args:
        model_wrapper (Model): The model wrapper.
        dataloader (DataLoader): DataLoader for the input data.
        max_new_tokens (int): The maximum number of new tokens to generate.
        sys_prompt_obf (torch.Tensor): The obfuscated system prompt (IDs or embedding).
        original_sys_prompt_ids (torch.Tensor): Token IDs of the placeholder system prompt to be replaced.
        is_soft_prompt_method (bool): True if `sys_prompt_obf` is an embedding, False if it's token IDs.
        obf_sys_prompt_len (int): The length of the obfuscated prompt.
        pad_token_id (int): The ID of the padding token.

    Returns:
        (tuple[torch.Tensor, torch.Tensor, int]): Tensors for log probabilities, generated IDs, and max generated length.
    """
    probs_list = []
    ids_list = []
    device = model_wrapper.device
    gpu_memory_used = []
    max_generated_length = 0
    total_samples_processed = 0
    for data_batch in tqdm(dataloader, desc="Precomputing model outputs with replaced system prompt"):
        input_batch = data_batch['input_ids']
        attention_mask_batch = data_batch['attention_mask']
        current_batch_size = input_batch.shape[0]

        sys_prompt_indices_batch = find_sys_prompt_indices_batch(
            input_batch, original_sys_prompt_ids, 
            pad_token_id, model_wrapper.name_or_path
        )

        if is_soft_prompt_method:
            input_batch = model_wrapper.get_embeddings(input_batch)

        input_batch = replace_sys_prompt_batch(
            sys_prompt_obf, input_batch, sys_prompt_indices_batch
        )

        attention_mask = update_attention_mask_batch(
            obf_sys_prompt_len, attention_mask_batch, sys_prompt_indices_batch
        )

        outputs = model_wrapper.generate_logits(
            input_tensor=input_batch.to(device, non_blocking=True),
            attention_mask=attention_mask.to(device, non_blocking=True),
            token_count=max_new_tokens,
            embedded=is_soft_prompt_method
        )

        batch_logits = torch.stack(outputs.scores, dim=0).detach().cpu()
        gen_len_this_batch = batch_logits.shape[0]
        if gen_len_this_batch > max_generated_length:
            max_generated_length = gen_len_this_batch
        batch_probs = torch.nn.functional.log_softmax(batch_logits, dim=-1)
        if is_soft_prompt_method:
            batch_generated_ids = outputs.sequences.detach().cpu()
        else:
            batch_generated_ids = outputs.sequences[:, input_batch.shape[1]:].detach().cpu()
        probs_list.append(batch_probs)
        ids_list.append(batch_generated_ids)

        if torch.cuda.is_available():
            gpu_memory_used.append(get_gpu_utilization())
    
        total_samples_processed += current_batch_size

    if gpu_memory_used:
        logger.info(f'Max GPU memory occupied during precomputation: {np.max(gpu_memory_used)//1024**2} MB')
    else:
        logger.info('GPU memory usage not tracked (either no CUDA or no batches processed).')

    final_probs_tensor = torch.zeros(
        (max_generated_length, total_samples_processed, model_wrapper.vocab_size),
        dtype=probs_list[0].dtype
    )
    final_ids_tensor = torch.full(
        (max_generated_length, total_samples_processed),
        model_wrapper.tokenizer.pad_token_id,
        dtype=ids_list[0].dtype
    )

    # Reshape the probs and ids tensors
    current_sample_idx = 0
    for i in range(len(probs_list)):
        batch_p = probs_list[i]
        batch_i = ids_list[i]
        gen_len_b = batch_p.shape[0]
        batch_sz_b = batch_p.shape[1]

        final_probs_tensor[:gen_len_b, current_sample_idx : current_sample_idx + batch_sz_b, :] = batch_p
        final_ids_tensor[:gen_len_b, current_sample_idx : current_sample_idx + batch_sz_b] = batch_i.transpose(0, 1)
        
        current_sample_idx += batch_sz_b

    return final_probs_tensor, final_ids_tensor, max_generated_length


def generate_model_responses(
    model_wrapper: Model,
    dataloader: DataLoader,
    generation_args: dict,
) -> list[list[str]]:
    """
    Generates text responses for a dataset using sampling.

    Args:
        model_wrapper (Model): The model wrapper.
        dataloader (DataLoader): DataLoader for the input data.
        generation_args (dict): A dictionary of arguments for sampling-based generation.

    Returns:
        (list[list[str]]): A list where each element is a list of generated string responses
                           for a corresponding input sample.
    """
    outputs = []
    gpu_memory_used = []
    device = model_wrapper.device
    for batch_data in tqdm(dataloader, desc="Generating responses"):
        input_ids_batch = batch_data['input_ids']
        attention_mask_batch = batch_data['attention_mask']

        model_output = model_wrapper.generate_output(
            input_tensor=input_ids_batch.to(device, non_blocking=True),
            attention_mask=attention_mask_batch.to(device, non_blocking=True),
            generation_params=generation_args,
            embedded=False
        )
        
        response_start_index = input_ids_batch.shape[1]
        decoded_outputs = []
        for output_list in model_output:
            output_list = output_list[:, response_start_index:]
            decoded_outputs.append(model_wrapper.tokenizer.batch_decode(output_list, skip_special_tokens=True))
        
        outputs.extend(decoded_outputs)

        if torch.cuda.is_available():
            gpu_memory_used.append(get_gpu_utilization())
    
    if gpu_memory_used:
        logger.info(f'Max GPU memory occupied during output generation: {np.max(gpu_memory_used)//1024**2} MB')
    else:
        logger.info('GPU memory usage not tracked for output generation (no CUDA).')
    return outputs


def generate_model_responses_replace(
    model_wrapper: Model,
    dataloader: DataLoader,
    generation_args: dict,
    sys_prompt_obf: torch.Tensor,
    original_sys_prompt_ids: torch.Tensor,
    is_soft_prompt_method: bool,
    obf_sys_prompt_len: int,
    pad_token_id: int,
) -> list[list[str]]:
    """

    Generates text responses after replacing the system prompt with an obfuscated version.

    Args:
        model_wrapper (Model): The model wrapper.
        dataloader (DataLoader): DataLoader for the input data.
        generation_args (dict): Dictionary of parameters for sampling-based generation.
        sys_prompt_obf (torch.Tensor): The obfuscated system prompt (IDs or embedding).
        original_sys_prompt_ids (torch.Tensor): Token IDs of the placeholder prompt to be replaced.
        is_soft_prompt_method (bool): True if `sys_prompt_obf` is an embedding.
        obf_sys_prompt_len (int): The length of the obfuscated prompt.
        pad_token_id (int): The ID of the padding token.

    Returns:
        (list[list[str]]): A list of generated string responses.
    """
    outputs = []
    gpu_memory_used = []
    device = model_wrapper.device
    for batch_data in tqdm(dataloader, desc="Generating responses with replaced system prompt"):
        input_batch = batch_data['input_ids']
        attention_mask_batch = batch_data['attention_mask']

        sys_prompt_indices_batch = find_sys_prompt_indices_batch(
            input_batch, original_sys_prompt_ids, 
            pad_token_id, model_wrapper.name_or_path
        )

        if is_soft_prompt_method:
            input_batch = model_wrapper.get_embeddings(input_batch)
        
        input_batch = replace_sys_prompt_batch(
            sys_prompt_obf, input_batch, sys_prompt_indices_batch
        )

        attention_mask = update_attention_mask_batch(
            obf_sys_prompt_len, attention_mask_batch, sys_prompt_indices_batch
        )

        model_output = model_wrapper.generate_output(
            input_tensor=input_batch.to(device, non_blocking=True),
            attention_mask=attention_mask.to(device, non_blocking=True),
            generation_params=generation_args,
            embedded=is_soft_prompt_method
        )

        # The transformers `generate` method behaves differently for token ID vs. embedding inputs.
        response_start_index = input_batch.shape[1]
        decoded_outputs = []
        for output_list in model_output:
            if not is_soft_prompt_method:
                # CASE 1: Hard Prompt (input_ids).
                # The model returns the full sequence (prompt + response).
                # We must slice off the prompt to isolate the response.
                output_list = output_list[:, response_start_index:]
            # else:
            #   # CASE 2: Soft Prompt (inputs_embeds).
            #   # The model returns *only* the generated response tokens.
            #   # No slicing is necessary. The 'output_list' is already the response.
            decoded_outputs.append(model_wrapper.tokenizer.batch_decode(output_list, skip_special_tokens=True))
        
        outputs.extend(decoded_outputs)

        if torch.cuda.is_available():
            gpu_memory_used.append(get_gpu_utilization())

    if gpu_memory_used:
        logger.info(f'Max GPU memory occupied during obfuscated output generation: {np.max(gpu_memory_used)//1024**2} MB')
    else:
        logger.info('GPU memory usage not tracked for obfuscated output generation (no CUDA).')
    return outputs


