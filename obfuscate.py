import json
import logging
import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader import load_and_prepare_dataset
from data.utils import TextDataset, create_collate_fn
from src.logging_config import setup_logging
from src.model import Model
from src.output_generation import precompute_model_outputs
from src.prompt_utils import *
from src.style_prompts import get_style_prompt
from src.utils import *

console = Console()

def get_args() -> Namespace:
    """Parses and validates command-line arguments for the system prompt obfuscation script."""
    parser = ArgumentParser(
        description="Script for system prompt obfuscation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Huggingface model name to use for optimization"
    )
    quantization_group = parser.add_mutually_exclusive_group(required=False)
    quantization_group.add_argument(
        "--quantize_4bit",
        action="store_true",
        default=True,
        help="Enable 4-bit quantization for the model. (Cannot be used with --quantize_8bit)"
    )
    quantization_group.add_argument(
        "--quantize_8bit",
        action="store_true",
        default=False,
        help="Enable 8-bit quantization for the model. (Cannot be used with --quantize_4bit)"
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Specify a custom system prompt directly as a string."
    )
    prompt_group.add_argument(
        "--style",
        type=str,
        default=None,
        help=(
            "Specify a predefined style for the system prompt.\n"
            "The available styles are defined in 'style_prompts.py'."
        )
    )
    parser.add_argument(
        "--obfuscation_method",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Method for obfuscating the system prompt"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for optimization"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=800,
        help="Dataset size for optimization (80:20 split)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="truthfulqa",
        choices = ["truthfulqa", "triviaqa", "cnn_dailymail", "samsum"],
        help="Dataset to use for optimization"    
    )
    parser.add_argument(
        "--task_hints",
        default=False,
        action="store_true",
        help="Whether to use task hints"
    )
    parser.add_argument(
        "--obf_sys_prompt_len",
        type=int,
        default=10,
        help="Length of the randomly initialized obfuscated system prompt in tokens."
    )
    parser.add_argument(
        "--output_token_count",
        type=int,
        default=15,
        help="Number of output tokens to optimize over"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Number of tokens in the context window to consider for gradient calculation"
    )
    parser.add_argument(
        "--optimizer_iter",
        type=int,
        default=10,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for optimization (only used for soft prompt obfuscation)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="topk value for GCG (only used for hard prompt obfuscation)"
    )
    parser.add_argument(
        "--search_width",
        type=int,
        default=10,
        help="search_width value for GCG (only used for hard prompt obfuscation)"
    )
    parser.add_argument(
        "--n_replace",
        type=int,
        default=1,
        help="n_replace value for GCG (only used for hard prompt obfuscation)"
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="Weight for cross-entropy loss"
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help="Weight for KL divergence loss"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/obfuscation",
        help="Output directory for obfuscation results"
    )
    args = parser.parse_args()

    return args

def obfuscate_soft_prompt(
    model_wrapper: Model,
    precomputed_probs: torch.Tensor,
    precomputed_ids: torch.Tensor,
    train_dataloader_conventional: DataLoader,
    sys_prompt_obf: torch.Tensor,
    original_sys_prompt_ids: torch.Tensor,
    obf_sys_prompt_len: int,
    lr: float,
    optimizer_iter: int,
    token_windows: list[list[int]],
    ce_weight: float,
    kl_weight: float,
) -> tuple[list[torch.Tensor], list[float]]:
    """
    Obfuscates the system prompt in the continuous embedding space (soft prompt).

    This function optimizes an initial random embedding (`sys_prompt_obf_emb`) to produce
    the same output distributions as the original system prompt for a given dataset.
    The optimization is done iteratively over windows of the target output sequence.

    Args:
        model_wrapper (Model): The model wrapper instance.
        precomputed_probs (torch.Tensor): Precomputed target probabilities from the original prompt.
        precomputed_ids (torch.Tensor): Precomputed target token IDs from the original prompt.
        train_dataloader_conventional (DataLoader): DataLoader providing inputs formatted with the original prompt.
        sys_prompt_obf (torch.Tensor): The initial random token IDs for the obfuscated prompt.
        original_sys_prompt_ids (torch.Tensor): Token IDs of the original system prompt, used for replacement.
        obf_sys_prompt_len (int): The length of the obfuscated prompt in tokens.
        lr (float): Learning rate for the Adam optimizer.
        optimizer_iter (int): Number of optimization iterations per window.
        token_windows (list[list[int]]): A list of token windows to optimize over.
        ce_weight (float): Weight for the cross-entropy loss.
        kl_weight (float): Weight for the KL divergence loss.

    Returns:
        (tuple[list[torch.Tensor], list[float]]): A tuple containing:
            - A list of the optimized soft prompt embeddings after each iteration.
            - A list of the training loss at each step.
    """
    sys_prompt_obf_emb = model_wrapper.get_embeddings(sys_prompt_obf).detach()
    sys_prompt_obf_emb = sys_prompt_obf_emb.requires_grad_(True)
    optimizer = torch.optim.Adam([sys_prompt_obf_emb], eps=1e-3, lr=lr)

    history_train_loss_per_iteration = []
    pad_token_id = model_wrapper.tokenizer.pad_token_id
    sys_prompt_emb_list = []

    total_train_samples = precomputed_ids.shape[1]
    # To store the cumulative true token IDs from previous completed windows for each sample
    cumulative_true_ids_for_dataset = torch.empty((total_train_samples, 0), dtype=precomputed_ids.dtype)

    cumulative_tokens_offset = 0 # Keeps track of how many tokens have been processed from previous windows

    for token_window_idx, token_window_indices in enumerate(token_windows):
        console.rule(f"[bold cyan]Token Window {token_window_idx + 1}/{len(token_windows)} (Output Tokens {min(token_window_indices)+1}-{max(token_window_indices)+1})", align="center")
        # Get the target probabilities and IDs for the current window
        current_window_target_probs_full = precomputed_probs[token_window_indices, :, :]
        current_window_target_ids_full = precomputed_ids[token_window_indices, :]
        num_tokens_in_window = len(token_window_indices)

        for iteration in range(optimizer_iter):
            logger.info(f'Token Window {token_window_idx + 1}, Iteration: {iteration + 1}/{optimizer_iter}')

            iteration_accumulated_loss_scalar = 0.0
            num_batches_processed = 0
            current_sample_offset_in_dataset = 0
            gpu_memory_used = []

            for batch_idx, data_batch in tqdm(enumerate(train_dataloader_conventional), 
                                              desc=f"Win {token_window_idx+1} Iter {iteration+1}", 
                                              total=len(train_dataloader_conventional)):
                optimizer.zero_grad()

                input_ids_batch = data_batch['input_ids']
                attention_mask_batch = data_batch['attention_mask']
                current_batch_actual_size = input_ids_batch.shape[0]

                # Get the indices of the system prompt in the current batch
                sys_prompt_indices_batch = find_sys_prompt_indices_batch(
                    input_ids_batch, original_sys_prompt_ids, 
                    pad_token_id, model_wrapper.name_or_path
                )
                # Replace the system prompt with the obfuscated version
                base_embedded_input_ids = model_wrapper.get_embeddings(input_ids_batch)
                base_embedded_input_ids = replace_sys_prompt_batch(
                    sys_prompt_obf_emb, base_embedded_input_ids, sys_prompt_indices_batch
                )
                # Update the attention mask to fit the obfuscated system prompt
                base_attention_mask = update_attention_mask_batch(
                    obf_sys_prompt_len, attention_mask_batch, sys_prompt_indices_batch
                )

                #Append cumulative true tokens from PREVIOUS windows
                cumulative_true_ids_for_batch = cumulative_true_ids_for_dataset[
                    current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size, :
                ]

                current_embedded_input_ids_batch = base_embedded_input_ids
                current_attention_mask_batch = base_attention_mask

                #If there are previous tokens
                if cumulative_true_ids_for_batch.shape[1] > 0:
                    # Append cumulative true tokens from PREVIOUS windows
                    embedded_cumulative_tokens = model_wrapper.get_embeddings(cumulative_true_ids_for_batch)
                    current_embedded_input_ids_batch = torch.cat(
                        [current_embedded_input_ids_batch, embedded_cumulative_tokens], dim=1
                    )
                    # Append cumulative attention mask
                    attention_for_cumulative_tokens = torch.ones_like(embedded_cumulative_tokens[..., 0], dtype=current_attention_mask_batch.dtype)
                    current_attention_mask_batch = torch.cat(
                        [current_attention_mask_batch, attention_for_cumulative_tokens], dim=1
                    )

                accumulated_loss_for_window_batch_tensor = torch.tensor(0.0, requires_grad=False)
                batch_target_probs_for_window_slice = current_window_target_probs_full[:, current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size, :]
                batch_target_ids_for_window_slice = current_window_target_ids_full[:, current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size]

                temp_embedded_inputs = current_embedded_input_ids_batch.clone()
                temp_attention_mask = current_attention_mask_batch.clone()

                logits_last = None
                next_token_logits_last = None
                next_token_log_probs_last = None

                # Calculate the loss for each token in the window
                for token_step_idx in range(num_tokens_in_window):
                    logits_last  = model_wrapper.model(
                        inputs_embeds=temp_embedded_inputs.to(model_wrapper.device, non_blocking=True),
                        attention_mask=temp_attention_mask.to(model_wrapper.device, non_blocking=True),
                    ).logits.float().cpu()

                    next_token_logits_last = logits_last[:, -1, :]
                    next_token_log_probs_last = F.log_softmax(next_token_logits_last, dim=-1)

                    true_log_probs_for_token = batch_target_probs_for_window_slice[token_step_idx, :, :]
                    true_ids_for_token = batch_target_ids_for_window_slice[token_step_idx, :]

                    loss_for_token_step = loss_function_with_padding_mask(
                        pred_logits=next_token_logits_last, pred_log_probs=next_token_log_probs_last,
                        true_log_probs=true_log_probs_for_token, true_ids=true_ids_for_token,
                        kl_weight=kl_weight, ce_weight=ce_weight, pad_token_id=pad_token_id
                    )
                    accumulated_loss_for_window_batch_tensor += loss_for_token_step

                    # Update the input and attention mask for the next token
                    true_next_token_embeddings = model_wrapper.get_embeddings(true_ids_for_token)
                    temp_embedded_inputs = torch.cat(
                        [temp_embedded_inputs, true_next_token_embeddings.unsqueeze(1)], dim=1
                    )
                    attention_for_new_token = torch.ones(
                        (current_batch_actual_size, 1), dtype=temp_attention_mask.dtype
                    )
                    temp_attention_mask = torch.cat([temp_attention_mask, attention_for_new_token], dim=1)

                avg_loss_for_batch_window_tensor = accumulated_loss_for_window_batch_tensor / num_tokens_in_window

                avg_loss_for_batch_window_tensor.backward()
                optimizer.step()

                iteration_accumulated_loss_scalar += avg_loss_for_batch_window_tensor.item()
                num_batches_processed += 1
                current_sample_offset_in_dataset += current_batch_actual_size

                gpu_memory_used.append(get_gpu_utilization())

                del input_ids_batch, attention_mask_batch, sys_prompt_indices_batch
                del base_embedded_input_ids, base_attention_mask
                del cumulative_true_ids_for_batch
                if 'embedded_cumulative_tokens' in locals(): del embedded_cumulative_tokens
                if 'attention_for_cumulative_tokens' in locals(): del attention_for_cumulative_tokens
                del current_embedded_input_ids_batch, current_attention_mask_batch
                del temp_embedded_inputs, temp_attention_mask
                del batch_target_probs_for_window_slice, batch_target_ids_for_window_slice
                del accumulated_loss_for_window_batch_tensor, avg_loss_for_batch_window_tensor
                if logits_last is not None: del logits_last
                if next_token_logits_last is not None: del next_token_logits_last
                if next_token_log_probs_last is not None: del next_token_log_probs_last
            

            avg_iteration_loss = iteration_accumulated_loss_scalar / num_batches_processed if num_batches_processed > 0 else 0.0
            logger.info(f'Token Window {token_window_idx + 1}, Iteration {iteration + 1} Avg Loss: {avg_iteration_loss:.4f}')
            history_train_loss_per_iteration.append(avg_iteration_loss)


            logger.info(f"Max GPU Utilization: {np.max(gpu_memory_used)//1024**2} MB")

            sys_prompt_emb_list.append(sys_prompt_obf_emb.clone().detach().cpu())
                

        # Append the true token IDs for the current window to the cumulative true token IDs
        true_ids_this_window_transposed = current_window_target_ids_full.transpose(0, 1)

        cumulative_true_ids_for_dataset = torch.cat(
            [cumulative_true_ids_for_dataset, true_ids_this_window_transposed], dim=1
        )
        cumulative_tokens_offset += num_tokens_in_window
        logger.info(f"Finished token window {token_window_idx + 1}. Cumulative true tokens appended: {num_tokens_in_window}. Total cumulative: {cumulative_tokens_offset}")
        del current_window_target_probs_full, current_window_target_ids_full, true_ids_this_window_transposed

    logger.info("Soft prompt obfuscation finished.")
            
    return sys_prompt_emb_list, history_train_loss_per_iteration




def obfuscate_hard_prompt(
    model_wrapper: Model,
    precomputed_probs: torch.Tensor,
    precomputed_ids: torch.Tensor,
    train_dataloader_conventional: DataLoader,
    sys_prompt_obf: torch.Tensor,
    original_sys_prompt_ids: torch.Tensor,
    obf_sys_prompt_len: int,
    optimizer_iter: int,
    token_windows: list[list[int]],
    topk: int,
    search_width: int,
    n_replace: int,
    ce_weight: float,
    kl_weight: float,
) -> tuple[list[torch.Tensor], list[float]]:
    """
    Obfuscates the system prompt in the discrete token space (hard prompt) using GCG.

    This function optimizes an initial random token sequence to produce the same output
    distributions as the original prompt. It uses the Greedy Coordinate Gradient (GCG)
    algorithm to find token replacements that minimize a combined CE and KL-divergence loss.

    Args:
        model_wrapper (Model): The model wrapper instance.
        precomputed_probs (torch.Tensor): Precomputed target probabilities from the original prompt.
        precomputed_ids (torch.Tensor): Precomputed target token IDs from the original prompt.
        train_dataloader_conventional (DataLoader): DataLoader for the dataset.
        sys_prompt_obf (torch.Tensor): The initial random token IDs for the obfuscated prompt.
        original_sys_prompt_ids (torch.Tensor): Token IDs of the original prompt for replacement.
        obf_sys_prompt_len (int): The length of the obfuscated prompt.
        optimizer_iter (int): Number of optimization iterations per window.
        token_windows (list[list[int]]): List of token windows to optimize over.
        topk (int): The 'k' in top-k for GCG candidate token selection.
        search_width (int): The number of candidate prompts to evaluate at each GCG step.
        n_replace (int): The number of tokens to replace at each GCG step.
        ce_weight (float): Weight for the cross-entropy loss.
        kl_weight (float): Weight for the KL divergence loss.

    Returns:
        (tuple[list[torch.Tensor], list[float]]): A tuple containing:
            - A list of the optimized hard prompt token ID tensors after each iteration.
            - A list of the training loss at each step.
    """
    embedding_layer_matrix = model_wrapper.get_embedding_matrix()
    history_train_loss_per_iteration = []
    pad_token_id = model_wrapper.tokenizer.pad_token_id
    sys_prompt_list = []
    vocab_size = model_wrapper.vocab_size

    total_train_samples = precomputed_ids.shape[1]
    # To store the cumulative true token IDs from previous completed windows for each sample
    cumulative_true_ids_for_dataset = torch.empty((total_train_samples, 0), dtype=precomputed_ids.dtype)

    cumulative_tokens_offset = 0 # Keeps track of how many tokens have been processed from previous windows

    device = model_wrapper.device

    for token_window_idx, token_window_indices in enumerate(token_windows):
        console.rule(f"[bold cyan]Token Window {token_window_idx + 1}/{len(token_windows)} (Output Tokens {min(token_window_indices)+1}-{max(token_window_indices)+1})", align="center")
        # Get the target probabilities and IDs for the current window
        current_window_target_probs_full = precomputed_probs[token_window_indices, :, :]
        current_window_target_ids_full = precomputed_ids[token_window_indices, :]
        num_tokens_in_window = len(token_window_indices)

        for iteration in range(optimizer_iter):
            logger.info(f'Token Window {token_window_idx + 1}, Iteration: {iteration + 1}/{optimizer_iter}')

            iteration_accumulated_loss_scalar = 0.0
            num_batches_processed = 0
            current_sample_offset_in_dataset = 0
            gpu_memory_used = []

            for batch_idx, data_batch in tqdm(enumerate(train_dataloader_conventional), 
                                              desc=f"Win {token_window_idx+1} Iter {iteration+1}", 
                                              total=len(train_dataloader_conventional)):
                input_ids_batch = data_batch['input_ids']
                attention_mask_batch = data_batch['attention_mask']
                current_batch_actual_size = input_ids_batch.shape[0]

                # Get the indices of the system prompt in the current batch
                sys_prompt_indices_batch = find_sys_prompt_indices_batch(
                    input_ids_batch, original_sys_prompt_ids, 
                    pad_token_id, model_wrapper.name_or_path
                )
                base_embedded_input_ids = model_wrapper.get_embeddings(input_ids_batch)

                # Create the obfuscated system prompt for GCG
                sys_prompt_obf_onehot = torch.nn.functional.one_hot(
                    sys_prompt_obf, 
                    num_classes=vocab_size
                )
                sys_prompt_obf_onehot = sys_prompt_obf_onehot.to(device, model_wrapper.dtype)
                sys_prompt_obf_onehot.requires_grad_()
                sys_prompt_obf_emb = sys_prompt_obf_onehot @ embedding_layer_matrix

                # Replace the system prompt in the current batch
                obf_embedded_input_ids = replace_sys_prompt_batch(
                    sys_prompt_obf_emb, 
                    base_embedded_input_ids.to(device, non_blocking=True),
                    sys_prompt_indices_batch
                ).cpu()
                # Update the attention mask to fit the obfuscated system prompt
                base_attention_mask = update_attention_mask_batch(
                    obf_sys_prompt_len, attention_mask_batch, sys_prompt_indices_batch
                )

                #Append cumulative true tokens from PREVIOUS windows
                cumulative_true_ids_for_batch = cumulative_true_ids_for_dataset[
                    current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size, :
                ]
                current_embedded_input_ids_batch = obf_embedded_input_ids
                current_attention_mask_batch = base_attention_mask

                #If there are previous tokens
                if cumulative_true_ids_for_batch.shape[1] > 0:
                    embedded_cumulative_tokens = model_wrapper.get_embeddings(cumulative_true_ids_for_batch)
                    # Append cumulative true tokens from PREVIOUS windows
                    current_embedded_input_ids_batch = torch.cat(
                        [current_embedded_input_ids_batch, embedded_cumulative_tokens], dim=1
                    )
                    attention_for_cumulative_tokens = torch.ones_like(embedded_cumulative_tokens[..., 0], dtype=current_attention_mask_batch.dtype)
                    # Append cumulative attention mask
                    current_attention_mask_batch = torch.cat(
                        [current_attention_mask_batch, attention_for_cumulative_tokens], dim=1
                    )
                
                accumulated_loss_for_window_batch_tensor = torch.tensor(0.0, requires_grad=False)
                batch_target_probs_for_window_slice = current_window_target_probs_full[:, current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size, :]
                batch_target_ids_for_window_slice = current_window_target_ids_full[:, current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size]

                temp_embedded_inputs = current_embedded_input_ids_batch.clone()
                temp_attention_mask = current_attention_mask_batch.clone()

                logits_last = None
                next_token_logits_last = None
                next_token_log_probs_last = None

                # Calculate the loss for each token in the window
                for token_step_idx in range(num_tokens_in_window):
                    logits_last  = model_wrapper.model(
                        inputs_embeds=temp_embedded_inputs.to(device, non_blocking=True),
                        attention_mask=temp_attention_mask.to(device, non_blocking=True),
                    ).logits.float().cpu()

                    next_token_logits_last = logits_last[:, -1, :]
                    next_token_log_probs_last = F.log_softmax(next_token_logits_last, dim=-1)

                    true_log_probs_for_token = batch_target_probs_for_window_slice[token_step_idx, :, :]
                    true_ids_for_token = batch_target_ids_for_window_slice[token_step_idx, :]

                    loss_for_token_step = loss_function_with_padding_mask(
                        pred_logits=next_token_logits_last, pred_log_probs=next_token_log_probs_last,
                        true_log_probs=true_log_probs_for_token, true_ids=true_ids_for_token,
                        kl_weight=kl_weight, ce_weight=ce_weight, pad_token_id=pad_token_id
                    )
                    accumulated_loss_for_window_batch_tensor += loss_for_token_step

                    true_next_token_embeddings = model_wrapper.get_embeddings(true_ids_for_token)
                    temp_embedded_inputs = torch.cat(
                        [temp_embedded_inputs, true_next_token_embeddings.unsqueeze(1)], dim=1
                    )
                    attention_for_new_token = torch.ones(
                        (current_batch_actual_size, 1), dtype=temp_attention_mask.dtype
                    )
                    temp_attention_mask = torch.cat([temp_attention_mask, attention_for_new_token], dim=1)

                avg_loss_for_batch_window_tensor = accumulated_loss_for_window_batch_tensor / num_tokens_in_window
                avg_loss_for_batch_window_tensor.backward()

                grad = sys_prompt_obf_onehot.grad.clone()
                # Get replacement candidates for the obfuscated system prompt
                candidates = get_GCG_candidates(
                    sys_prompt_obf,
                    grad,
                    search_width,
                    topk,
                    n_replace
                )
                candidate_losses = []
                # Recalculate the loss for each candidate
                for cand_idx, candidate_ids in enumerate(candidates):
                    embedded_candidate_ids = model_wrapper.get_embeddings(candidate_ids).cpu()

                    obf_embedded_input_ids_candidate = replace_sys_prompt_batch(
                        embedded_candidate_ids, 
                        base_embedded_input_ids,
                        sys_prompt_indices_batch
                    )

                    current_embedded_input_ids_batch = obf_embedded_input_ids_candidate
                    current_attention_mask_batch = base_attention_mask

                    if cumulative_true_ids_for_batch.shape[1] > 0:
                        embedded_cumulative_tokens = model_wrapper.get_embeddings(cumulative_true_ids_for_batch).cpu()
                        current_embedded_input_ids_batch = torch.cat(
                            [current_embedded_input_ids_batch, embedded_cumulative_tokens], dim=1
                        )
                        attention_for_cumulative_tokens = torch.ones_like(embedded_cumulative_tokens[..., 0], dtype=current_attention_mask_batch.dtype)
                        current_attention_mask_batch = torch.cat(
                            [current_attention_mask_batch, attention_for_cumulative_tokens], dim=1
                        )
                    
                    accumulated_loss_for_window_batch_tensor = torch.tensor(0.0, requires_grad=False)
                    temp_embedded_inputs = current_embedded_input_ids_batch.clone()
                    temp_attention_mask = current_attention_mask_batch.clone()

                    logits_last = None
                    next_token_logits_last = None
                    next_token_log_probs_last = None

                    for token_step_idx in range(num_tokens_in_window):
                        with torch.no_grad():
                            logits_last  = model_wrapper.model(
                                inputs_embeds=temp_embedded_inputs.to(device, non_blocking=True),
                                attention_mask=temp_attention_mask.to(device, non_blocking=True),
                            ).logits.float().cpu()

                        next_token_logits_last = logits_last[:, -1, :]
                        next_token_log_probs_last = F.log_softmax(next_token_logits_last, dim=-1)

                        true_log_probs_for_token = batch_target_probs_for_window_slice[token_step_idx, :, :]
                        true_ids_for_token = batch_target_ids_for_window_slice[token_step_idx, :]

                        loss_for_token_step = loss_function_with_padding_mask(
                            pred_logits=next_token_logits_last, pred_log_probs=next_token_log_probs_last,
                            true_log_probs=true_log_probs_for_token, true_ids=true_ids_for_token,
                            kl_weight=kl_weight, ce_weight=ce_weight, pad_token_id=pad_token_id
                        )
                        accumulated_loss_for_window_batch_tensor += loss_for_token_step

                        true_next_token_embeddings = model_wrapper.get_embeddings(true_ids_for_token).cpu()
                        temp_embedded_inputs = torch.cat(
                            [temp_embedded_inputs, true_next_token_embeddings.unsqueeze(1)], dim=1
                        )
                        attention_for_new_token = torch.ones(
                            (current_batch_actual_size, 1), dtype=temp_attention_mask.dtype
                        )
                        temp_attention_mask = torch.cat([temp_attention_mask, attention_for_new_token], dim=1)
                    
                    avg_loss_for_batch_window_tensor = accumulated_loss_for_window_batch_tensor / num_tokens_in_window
                    candidate_losses.append(avg_loss_for_batch_window_tensor)

                    del embedded_candidate_ids, obf_embedded_input_ids_candidate, current_embedded_input_ids_batch
                    del current_attention_mask_batch, accumulated_loss_for_window_batch_tensor, temp_embedded_inputs
                    del temp_attention_mask, logits_last, next_token_logits_last, next_token_log_probs_last
                    del true_log_probs_for_token, true_ids_for_token, loss_for_token_step, avg_loss_for_batch_window_tensor
                    del true_next_token_embeddings, attention_for_new_token
                    
                # Select the candidate with the lowest loss
                best_candidate_idx = np.argmin(candidate_losses)
                sys_prompt_obf = candidates[best_candidate_idx]

                iteration_accumulated_loss_scalar += np.min(candidate_losses)
                num_batches_processed += 1
                current_sample_offset_in_dataset += current_batch_actual_size

                gpu_memory_used.append(get_gpu_utilization())

                del input_ids_batch, attention_mask_batch, sys_prompt_indices_batch
                del base_embedded_input_ids, base_attention_mask
                del cumulative_true_ids_for_batch
                if 'embedded_cumulative_tokens' in locals(): del embedded_cumulative_tokens
                if 'attention_for_cumulative_tokens' in locals(): del attention_for_cumulative_tokens
                del batch_target_probs_for_window_slice, batch_target_ids_for_window_slice

            avg_iteration_loss = iteration_accumulated_loss_scalar / num_batches_processed if num_batches_processed > 0 else 0.0
            logger.info(f'Token Window {token_window_idx + 1}, Iteration {iteration + 1} Avg Loss: {avg_iteration_loss:.4f}')
            history_train_loss_per_iteration.append(avg_iteration_loss)
            
            sys_prompt_obf_str = model_wrapper.tokenizer.decode(sys_prompt_obf)
            logger.info(f"Current obfuscated system prompt: {sys_prompt_obf_str}")

            logger.info(f"Max GPU Utilization: {np.max(gpu_memory_used)//1024**2} MB")
            sys_prompt_list.append(sys_prompt_obf)

        # Append the true token IDs for the current window to the cumulative true token IDs
        true_ids_this_window_transposed = current_window_target_ids_full.transpose(0, 1)

        cumulative_true_ids_for_dataset = torch.cat(
            [cumulative_true_ids_for_dataset, true_ids_this_window_transposed], dim=1
        )
        cumulative_tokens_offset += num_tokens_in_window
        logger.info(f"Finished token window {token_window_idx + 1}. Cumulative true tokens appended: {num_tokens_in_window}. Total cumulative: {cumulative_tokens_offset}")
        del current_window_target_probs_full, current_window_target_ids_full, true_ids_this_window_transposed

    logger.info("Hard prompt obfuscation finished.")
            
    return sys_prompt_list, history_train_loss_per_iteration
       

def main(
    model_name: str,
    quantize_4bit: bool,
    quantize_8bit: bool,
    system_prompt: str | None,
    style: str | None,
    obfuscation_method: str,
    batch_size: int,
    dataset_size: int,
    dataset_name: str,
    task_hints: bool,
    obf_sys_prompt_len: int,
    output_token_count: int,
    window_size: int,
    optimizer_iter: int,
    lr: float,
    topk: int,
    search_width: int,
    n_replace: int,
    ce_weight: float,
    kl_weight: float,
    seed: int,
    output_dir: str,
) -> None:
    """
    Main function for the system prompt obfuscation process.

    1.  Loads the model and a dataset for optimization.
    2.  Constructs the original (conventional) system prompt.
    3.  Precomputes the target model outputs (probabilities and token IDs) using the
        conventional prompt. These serve as the ground truth for the optimization.
    4.  Initializes a random prompt (either as token IDs or an embedding).
    5.  Runs the chosen obfuscation method ('soft' or 'hard') to optimize the random
        prompt to match the behavior of the conventional one.
    6.  Saves the resulting obfuscated prompt(s), training loss, prepared data,
        and all configuration parameters to the specified output directory.

    Args:
        model_name (str): Huggingface model name or path.
        quantize_4bit (bool): If True, use 4-bit quantization.
        quantize_8bit (bool): If True, use 8-bit quantization.
        system_prompt (str | None): A custom system prompt string.
        style (str | None): A predefined style name for the system prompt.
        obfuscation_method (str): The method to use: "soft" or "hard".
        batch_size (int): Batch size for optimization steps.
        dataset_size (int): Total number of samples to use from the dataset.
        dataset_name (str): Name of the dataset to use for optimization.
        task_hints (bool): If True, task-specific hints are used in user prompts.
        obf_sys_prompt_len (int): Length of the obfuscated prompt in tokens.
        output_token_count (int): Number of output tokens to optimize over.
        window_size (int): Size of the context window for gradient calculation.
        optimizer_iter (int): Number of optimization iterations.
        lr (float): Learning rate for soft prompt optimization.
        topk (int): topk value for GCG (hard prompt obfuscation).
        search_width (int): Search width for GCG (hard prompt obfuscation).
        n_replace (int): Number of token replacements for GCG.
        ce_weight (float): Weight for the cross-entropy loss.
        kl_weight (float): Weight for the KL divergence loss.
        seed (int): Seed for reproducibility.
        output_dir (str): Directory to save obfuscation results.
    """
    logger = logging.getLogger(__name__)

    if quantize_4bit:
        quantization_mode = "4bit"
    elif quantize_8bit:
        quantization_mode = "8bit"
    else:
        quantization_mode = None
    
    set_seed(seed)
    logger.info("Loading tokenizer and model...")
    try:
        model_wrapper = Model(model_name, quantization_mode)
    except Exception as e:
        logger.exception(f"Failed to load model '{model_name}'. Error: {e}")
        return

    logger.info("Loading dataset...")
    try:
        train_samples, test_samples, task_system_prompt = load_and_prepare_dataset(
            dataset_name,
            dataset_size,
            task_hints,
            seed,
            split_ratio=0.8
        )
    except ValueError as e:
        logger.error(f"Error during dataset preparation: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during dataset preparation for {dataset_name}.")
        sys.exit(1)
    
    logger.debug(f"Example training sample: {train_samples[0]}")

    logger.info("Constructing system prompt for obfuscation...")
    if system_prompt is not None:
        conventional_sys_prompt = system_prompt
        logger.debug(f"Using custom system prompt: '{conventional_sys_prompt}'")
    elif style is not None:
        conventional_sys_prompt = get_style_prompt(style)
        if not conventional_sys_prompt:
            logger.error(f"Style '{style}' not found in predefined styles (src/style_prompts.py). Exiting.")
            sys.exit(1)
        logger.debug(f"Using style prompt for '{style}': '{conventional_sys_prompt}'")
    
    pad_token_string = model_wrapper.tokenizer.pad_token

    if not task_hints:
        conventional_sys_prompt = f"{pad_token_string}{task_system_prompt} {conventional_sys_prompt}{pad_token_string}"
    else:
        conventional_sys_prompt = f"{pad_token_string}{conventional_sys_prompt}{pad_token_string}"

    conventional_sys_prompt_ids = model_wrapper.tokenizer(
        conventional_sys_prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0]
    
    logger.info(f"Constructed system prompt for obfuscation: '{conventional_sys_prompt}'")

    logger.info(f"Initializing obfuscated system prompt with length: {obf_sys_prompt_len} tokens.")
    sys_prompt_obf = generate_random_token_sequence(obf_sys_prompt_len, model_wrapper.vocab_size)
    decoded_initial_obf = model_wrapper.tokenizer.decode(sys_prompt_obf, skip_special_tokens=False)
    logger.debug(f'Initial obfuscated system prompt IDs: {sys_prompt_obf.tolist()}')
    logger.debug(f'Decoded initial obfuscated system prompt: {decoded_initial_obf}')

   
    train_dataset = TextDataset(train_samples)

    conventional_collate_fn = create_collate_fn(
        tokenizer=model_wrapper.tokenizer,
        system_prompt=conventional_sys_prompt
    )

    train_dataloader_conventional = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=conventional_collate_fn,
        shuffle=False
    )


    logger.info("Precomputing model outputs (probs and IDs) using conventional system prompt...")
    precomputed_probs, precomputed_ids, max_generated_length = precompute_model_outputs(
        model_wrapper=model_wrapper,
        dataloader=train_dataloader_conventional,
        max_new_tokens=output_token_count
    )

    output_token_count = max_generated_length
    if window_size >= output_token_count:
        window_size = output_token_count
    
    token_windows = create_non_overlapping_windows(output_token_count, window_size)
    logger.debug(f"Number of output token windows: {len(token_windows)}")
    
    if obfuscation_method == "soft":
        logger.info("Applying soft prompt obfuscation...")
        sys_prompt_obf_list, train_loss_per_iteration = obfuscate_soft_prompt(
            model_wrapper,
            precomputed_probs,
            precomputed_ids,
            train_dataloader_conventional,
            sys_prompt_obf,
            conventional_sys_prompt_ids,
            obf_sys_prompt_len,
            lr,
            optimizer_iter,
            token_windows,
            ce_weight,
            kl_weight
        )
    elif obfuscation_method == "hard":
        logger.info("Applying hard prompt obfuscation...")
        sys_prompt_obf_list, train_loss_per_iteration = obfuscate_hard_prompt(
            model_wrapper,
            precomputed_probs,
            precomputed_ids,
            train_dataloader_conventional,
            sys_prompt_obf,
            conventional_sys_prompt_ids,
            obf_sys_prompt_len,
            optimizer_iter,
            token_windows,
            topk,
            search_width,
            n_replace,
            ce_weight,
            kl_weight
        )
    
    params = {
        "model_name": model_name,
        "quantize_4bit": quantize_4bit,
        "quantize_8bit": quantize_8bit,
        "system_prompt": conventional_sys_prompt,
        "style": style,
        "obfuscation_method": obfuscation_method,
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "dataset_size": dataset_size,
        "task_hints": task_hints,
        "obf_sys_prompt_len": obf_sys_prompt_len,
        "output_token_count": output_token_count,
        "window_size": window_size,
        "lr": lr,
        "optimizer_iter": optimizer_iter,
        "topk": topk,
        "search_width": search_width,
        "n_replace": n_replace,
        "ce_weight": ce_weight,
        "kl_weight": kl_weight,
        "seed": seed
    }

    output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(output_dir_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    torch.save(sys_prompt_obf_list, output_dir_path / "obfuscated_system_prompt_list.pt")


    np.save(output_dir_path / "train_loss.npy", np.asarray(train_loss_per_iteration))


    # For prepared_data, also use the Path object
    prepared_data_dir = output_dir_path / "prepared_data"
    prepared_data_dir.mkdir(parents=True, exist_ok=True)

    train_texts_file = prepared_data_dir / "train_data.json"
    test_texts_file = prepared_data_dir / "test_data.json"

    with open(train_texts_file, "w") as f:
        json.dump(train_samples, f, indent=4)

    with open(test_texts_file, "w") as f:
        json.dump(test_samples, f, indent=4)

    logger.info(f"Results saved to {output_dir_path.resolve()}")

if __name__ == "__main__":
    setup_logging('obfuscate.log', 'INFO') # Change to 'DEBUG' for more verbose logging
    logger = logging.getLogger(__name__)
    
    logger.debug("Parsing command line arguments...")
    try:
        args = get_args()
        logger.info(f"Command line arguments received: {json.dumps(vars(args), indent=2)}")
        main(**vars(args))
    except SystemExit:
        logger.warning("Exiting due to argument parsing issue (e.g., --help or invalid arguments).")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An critical error occurred: {e}")
        sys.exit(1)
    finally:
        logger.info("Done.")