import json
import logging
import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader import load_and_prepare_dataset
from data.utils import TextDataset, create_collate_fn
from src.logging_config import setup_logging
from src.model import Model
from src.output_generation import precompute_model_outputs_replace
from src.prompt_utils import *
from src.utils import *

console = Console()

def get_args() -> Namespace:
    """Parses command-line arguments for the fluency deobfuscation script."""
    parser = ArgumentParser(
        description="Script for system prompt deobfuscation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where obfuscate.py saved its results."
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        required=True,
        help="Path to the tensor file containing embeddings."
    )
    parser.add_argument(
        "--deobfuscation_method",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Method for deobfuscating the embedded system prompt"
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
        default=5,
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
        "--consistency_loss_weight",
        type=float,
        default=1.0,
        help="Weight for consistency loss"
    )
    parser.add_argument(
        "--fluency_loss_weight",
        type=float,
        default=1.0,
        help="Weight for fluency loss"
    )
    parser.add_argument(
        "--deobfuscated_sys_prompts_filename",
        type=str,
        default="deobfuscated_sys_prompt_list.pt",
        help="Filename for the output file containing deobfuscated system prompts."
    )
    args = parser.parse_args()

    return args


def euclidean_projection(
    embedding_layer: torch.nn.Embedding, 
    prompt_embedding: torch.Tensor
) -> torch.Tensor:
    """Projects an embedding to the nearest token ID based on Euclidean distance."""
    with torch.no_grad():
        embedding_layer_weights = embedding_layer.weight.cpu()
        distances = torch.cdist(prompt_embedding.double(), embedding_layer_weights.double(), p=2)
        nearest_token_ids = torch.argmin(distances, dim=1)
    return nearest_token_ids

def euclidean_projection_differentiable(
    embedding_layer: torch.nn.Embedding,
    prompt_embedding: torch.Tensor,
    tau: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Projects an embedding to a differentiable one-hot vector using the Gumbel-Softmax trick.

    This allows gradients to flow back through the projection step during optimization.
    It returns both the "soft" embeddings (a weighted average based on softmax probabilities)
    and the "hard" token IDs from the argmax of the one-hot vector.

    Args:
        embedding_layer (torch.nn.Embedding): The model's word embedding layer.
        prompt_embedding (torch.Tensor): The soft prompt embedding to project.
        tau (float): The temperature parameter for the softmax function.

    Returns:
        (tuple[torch.Tensor, torch.Tensor]): A tuple containing:
            - The differentiable projected embeddings (soft).
            - The corresponding token IDs (hard).
    """
    embedding_layer_weights = embedding_layer.weight.cpu()
    distances = torch.cdist(prompt_embedding.double(), embedding_layer_weights.double(), p=2)

    # Use softmax as a differentiable approximation of argmin
    probabilities = F.softmax(-distances/tau, dim=0)

    # Straight-Through-Estimator: use one-hot for forward pass, but probabilities for backward pass
    index = probabilities.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(distances).scatter_(-1, index, 1.0)
    one_hot_indices = y_hard - probabilities.detach() + probabilities


    chosen_embeddings = one_hot_indices @ embedding_layer_weights.double()
    
    return chosen_embeddings.half(), torch.argmax(one_hot_indices, dim=1)


def deobfuscate_soft_prompt(
    model_wrapper: Model,
    precomputed_probs: torch.Tensor,
    precomputed_ids: torch.Tensor,
    train_dataloader: DataLoader,
    sys_prompt_obf: torch.Tensor,
    original_sys_prompt_ids: torch.Tensor,
    obf_sys_prompt_len: int,
    lr: float,
    optimizer_iter: int,
    token_windows: list[list[int]],
    ce_weight: float,
    kl_weight: float,
    consistency_loss_weight: float,
    fluency_loss_weight: float,
) -> tuple[list[torch.Tensor], list[float]]:
    """
    Deobfuscates a soft prompt by optimizing its embedding for fluency and consistency.

    This function attempts to recover a readable prompt by iteratively adjusting an
    embedding vector. The loss function has two main components:
    1.  Consistency Loss: Ensures the deobfuscated prompt produces similar outputs
        to the original obfuscated prompt (using precomputed targets).
    2.  Fluency Loss: Encourages the projection of the embedding to be a
        semantically coherent sequence of tokens (by minimizing its NLL).

    Args:
        model_wrapper (Model): The model wrapper instance.
        precomputed_probs (torch.Tensor): Precomputed target probabilities from the obfuscated prompt.
        precomputed_ids (torch.Tensor): Precomputed target token IDs from the obfuscated prompt.
        train_dataloader (DataLoader): DataLoader for the dataset used in the attack.
        sys_prompt_obf (torch.Tensor): The initial obfuscated soft prompt embedding.
        original_sys_prompt_ids (torch.Tensor): Token IDs of a placeholder prompt to find replacement indices.
        obf_sys_prompt_len (int): The length of the obfuscated prompt.
        lr (float): Learning rate for the Adam optimizer.
        optimizer_iter (int): Number of optimization iterations per window.
        token_windows (list[list[int]]): A list of token windows to optimize over.
        ce_weight (float): Weight for the cross-entropy part of the consistency loss.
        kl_weight (float): Weight for the KL-divergence part of the consistency loss.
        consistency_loss_weight (float): Overall weight for the consistency loss.
        fluency_loss_weight (float): Weight for the fluency (NLL) loss.

    Returns:
        (tuple[list[torch.Tensor], list[float]]): A tuple containing:
            - A list of the projected deobfuscated token IDs after each optimization step.
            - A list of the training loss at each step.
    """
    deobfuscated_sys_prompt = sys_prompt_obf.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([deobfuscated_sys_prompt], eps=1e-3, lr=lr)
    embedding_layer = model_wrapper.word_embedding_layer

    deobf_sys_prompt_attention_mask = torch.ones(obf_sys_prompt_len)

    history_train_loss_per_iteration = []
    pad_token_id = model_wrapper.tokenizer.pad_token_id
    deobf_sys_prompt_list = []
    total_train_samples = precomputed_ids.shape[1]
    # To store the cumulative true token IDs from previous completed windows for each sample
    cumulative_true_ids_for_dataset = torch.empty((total_train_samples, 0), dtype=precomputed_ids.dtype)

    cumulative_tokens_offset = 0 # Keeps track of how many tokens have been processed from previous windows
    projected_embeddings, projected_ids = euclidean_projection_differentiable(
        embedding_layer,
        deobfuscated_sys_prompt,
        1.0
    )
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

            for batch_idx, data_batch in tqdm(enumerate(train_dataloader), 
                                              desc=f"Win {token_window_idx+1} Iter {iteration+1}", 
                                              total=len(train_dataloader)):
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
                    deobfuscated_sys_prompt, base_embedded_input_ids, sys_prompt_indices_batch
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

                consistency_loss_for_window_batch_tensor = torch.tensor(0.0, requires_grad=False)
                batch_target_probs_for_window_slice = current_window_target_probs_full[:, current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size, :]
                batch_target_ids_for_window_slice = current_window_target_ids_full[:, current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size]

                temp_embedded_inputs = current_embedded_input_ids_batch.clone()
                temp_attention_mask = current_attention_mask_batch.clone()

                logits_last = None
                next_token_logits_last = None
                next_token_log_probs_last = None

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
                    consistency_loss_for_window_batch_tensor += loss_for_token_step

                    # Update the input and attention mask for the next token
                    true_next_token_embeddings = model_wrapper.get_embeddings(true_ids_for_token)
                    temp_embedded_inputs = torch.cat(
                        [temp_embedded_inputs, true_next_token_embeddings.unsqueeze(1)], dim=1
                    )
                    attention_for_new_token = torch.ones(
                        (current_batch_actual_size, 1), dtype=temp_attention_mask.dtype
                    )
                    temp_attention_mask = torch.cat([temp_attention_mask, attention_for_new_token], dim=1)

                # Project the current obfuscated embedding back to token space
                projected_embeddings, projected_ids = euclidean_projection_differentiable(
                    embedding_layer,
                    deobfuscated_sys_prompt,
                    1.0
                )
                # Calculate NLL loss of the projected ids as an approximation for fluency
                output = model_wrapper.model(
                    inputs_embeds=projected_embeddings[None, :].to(device, non_blocking=True), 
                    attention_mask=deobf_sys_prompt_attention_mask.to(device, non_blocking=True),
                    labels = projected_ids.to(device, non_blocking=True)
                )
                fluency_loss_for_batch = output.loss

                avg_consistency_loss_for_batch_window_tensor = consistency_loss_for_window_batch_tensor / num_tokens_in_window
                combined_loss_for_batch = consistency_loss_weight * avg_consistency_loss_for_batch_window_tensor + fluency_loss_weight * fluency_loss_for_batch
                combined_loss_for_batch.backward()
                optimizer.step()

                iteration_accumulated_loss_scalar += combined_loss_for_batch.detach().cpu().item()
                num_batches_processed += 1
                current_sample_offset_in_dataset += current_batch_actual_size

                gpu_memory_used.append(get_gpu_utilization())

                del input_ids_batch, attention_mask_batch, sys_prompt_indices_batch
                del base_embedded_input_ids, base_attention_mask
                del cumulative_true_ids_for_batch
                del output, projected_embeddings, projected_ids, loss_for_token_step
                if 'embedded_cumulative_tokens' in locals(): del embedded_cumulative_tokens
                if 'attention_for_cumulative_tokens' in locals(): del attention_for_cumulative_tokens
                del current_embedded_input_ids_batch, current_attention_mask_batch
                del temp_embedded_inputs, temp_attention_mask
                del batch_target_probs_for_window_slice, batch_target_ids_for_window_slice
                del consistency_loss_for_window_batch_tensor, avg_consistency_loss_for_batch_window_tensor
                del combined_loss_for_batch, fluency_loss_for_batch
                if logits_last is not None: del logits_last
                if next_token_logits_last is not None: del next_token_logits_last
                if next_token_log_probs_last is not None: del next_token_log_probs_last
            
            avg_iteration_loss = iteration_accumulated_loss_scalar / num_batches_processed if num_batches_processed > 0 else 0.0
            logger.info(f'Token Window {token_window_idx + 1}, Iteration {iteration + 1} Avg Loss: {avg_iteration_loss:.4f}')
            history_train_loss_per_iteration.append(avg_iteration_loss)


            logger.info(f"Max GPU Utilization: {np.max(gpu_memory_used)//1024**2} MB")
            projected_embeddings, projected_ids = euclidean_projection_differentiable(
                embedding_layer,
                deobfuscated_sys_prompt,
                1.0
            )
            logger.info(f"Current deobfuscated system prompt: {model_wrapper.tokenizer.decode(projected_ids)}")
            deobf_sys_prompt_list.append(projected_ids.clone().detach().cpu())
        
        true_ids_this_window_transposed = current_window_target_ids_full.transpose(0, 1)

        cumulative_true_ids_for_dataset = torch.cat(
            [cumulative_true_ids_for_dataset, true_ids_this_window_transposed], dim=1
        )
        cumulative_tokens_offset += num_tokens_in_window
        logger.info(f"Finished token window {token_window_idx + 1}. Cumulative true tokens appended: {num_tokens_in_window}. Total cumulative: {cumulative_tokens_offset}")
        del current_window_target_probs_full, current_window_target_ids_full, true_ids_this_window_transposed
    
    logger.info("Soft prompt deobfuscation finished.")
    return deobf_sys_prompt_list, history_train_loss_per_iteration



def deobfuscate_hard_prompt(
    model_wrapper: Model,
    precomputed_probs: torch.Tensor,
    precomputed_ids: torch.Tensor,
    train_dataloader: DataLoader,
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
    consistency_loss_weight: float,
    fluency_loss_weight: float,
) -> tuple[list[torch.Tensor], list[float]]:
    """
    Deobfuscates a soft prompt by optimizing a discrete token sequence (hard prompt) for fluency and consistency.

    This function first projects the initial soft prompt into a hard prompt (a sequence of token IDs).
    It then uses the Greedy Coordinate Gradient (GCG) method to iteratively replace tokens in this
    hard prompt to minimize a combined loss function.

    The loss function has two main components:
    1.  Consistency Loss: Ensures the deobfuscated prompt produces similar outputs
        to the original obfuscated prompt (using precomputed targets).
    2.  Fluency Loss: Encourages the hard prompt to be a semantically coherent
        sequence of tokens (by minimizing its NLL).

    Args:
        model_wrapper (Model): The model wrapper instance.
        precomputed_probs (torch.Tensor): Precomputed target probabilities from the obfuscated prompt.
        precomputed_ids (torch.Tensor): Precomputed target token IDs from the obfuscated prompt.
        train_dataloader (DataLoader): DataLoader for the dataset used in the attack.
        sys_prompt_obf (torch.Tensor): The initial obfuscated soft prompt embedding.
        original_sys_prompt_ids (torch.Tensor): Token IDs of a placeholder prompt to find replacement indices.
        obf_sys_prompt_len (int): The length of the obfuscated prompt.
        optimizer_iter (int): Number of optimization iterations per window.
        token_windows (list[list[int]]): A list of token windows to optimize over.
        topk (int): The 'k' in top-k for GCG candidate token selection.
        search_width (int): The number of candidate prompts to evaluate at each GCG step.
        n_replace (int): The number of tokens to replace at each GCG step.
        ce_weight (float): Weight for the cross-entropy part of the consistency loss.
        kl_weight (float): Weight for the KL-divergence part of the consistency loss.
        consistency_loss_weight (float): Overall weight for the consistency loss.
        fluency_loss_weight (float): Weight for the fluency (NLL) loss.

    Returns:
        (tuple[list[torch.Tensor], list[float]]): A tuple containing:
            - A list of the deobfuscated hard prompt token ID tensors after each optimization step.
            - A list of the training loss at each step.
    """
    deobf_sys_prompt_ids = euclidean_projection(model_wrapper.word_embedding_layer, sys_prompt_obf)
    deobf_sys_prompt_attention_mask = torch.ones(obf_sys_prompt_len)

    embedding_layer_matrix = model_wrapper.get_embedding_matrix()
    history_train_loss_per_iteration = []
    pad_token_id = model_wrapper.tokenizer.pad_token_id
    deobf_sys_prompt_list = []
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

            for batch_idx, data_batch in tqdm(enumerate(train_dataloader), 
                                              desc=f"Win {token_window_idx+1} Iter {iteration+1}", 
                                              total=len(train_dataloader)):
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
                    deobf_sys_prompt_ids, 
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
                
                consistency_loss_for_window_batch_tensor = torch.tensor(0.0, requires_grad=False)
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
                    consistency_loss_for_window_batch_tensor += loss_for_token_step

                    true_next_token_embeddings = model_wrapper.get_embeddings(true_ids_for_token)
                    temp_embedded_inputs = torch.cat(
                        [temp_embedded_inputs, true_next_token_embeddings.unsqueeze(1)], dim=1
                    )
                    attention_for_new_token = torch.ones(
                        (current_batch_actual_size, 1), dtype=temp_attention_mask.dtype
                    )
                    temp_attention_mask = torch.cat([temp_attention_mask, attention_for_new_token], dim=1)
                
                # Calculate NLL loss of the projected ids as an approximation for fluency
                output = model_wrapper.model(
                    inputs_embeds=sys_prompt_obf_emb[None, :].to(device, non_blocking=True), 
                    attention_mask=deobf_sys_prompt_attention_mask.to(device, non_blocking=True),
                    labels = deobf_sys_prompt_ids.to(device, non_blocking=True)
                )
                fluency_loss_for_batch = output.loss

                avg_consistency_loss_for_batch_window_tensor = consistency_loss_for_window_batch_tensor / num_tokens_in_window
                combined_loss_for_batch = consistency_loss_weight * avg_consistency_loss_for_batch_window_tensor + fluency_loss_weight * fluency_loss_for_batch
                combined_loss_for_batch.backward()

                grad = sys_prompt_obf_onehot.grad.clone()
                # Get replacement candidates for the deobfuscated system prompt
                candidates = get_GCG_candidates(
                    deobf_sys_prompt_ids,
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
                    
                    consistency_loss_for_window_batch_tensor = torch.tensor(0.0, requires_grad=False)
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
                        consistency_loss_for_window_batch_tensor += loss_for_token_step

                        true_next_token_embeddings = model_wrapper.get_embeddings(true_ids_for_token).cpu()
                        temp_embedded_inputs = torch.cat(
                            [temp_embedded_inputs, true_next_token_embeddings.unsqueeze(1)], dim=1
                        )
                        attention_for_new_token = torch.ones(
                            (current_batch_actual_size, 1), dtype=temp_attention_mask.dtype
                        )
                        temp_attention_mask = torch.cat([temp_attention_mask, attention_for_new_token], dim=1)
                    
                    output = model_wrapper.model(
                        input_ids=candidate_ids[None, :].to(device, non_blocking=True), 
                        attention_mask=deobf_sys_prompt_attention_mask.to(device, non_blocking=True),
                        labels=candidate_ids.to(device, non_blocking=True)
                    )
                    fluency_loss_for_batch = output.loss

                    avg_consistency_loss_for_batch_window_tensor = consistency_loss_for_window_batch_tensor / num_tokens_in_window
                    combined_loss_for_batch = consistency_loss_weight * avg_consistency_loss_for_batch_window_tensor + fluency_loss_weight * fluency_loss_for_batch
                    candidate_losses.append(combined_loss_for_batch.detach().cpu())

                    del embedded_candidate_ids, obf_embedded_input_ids_candidate, current_embedded_input_ids_batch
                    del current_attention_mask_batch, consistency_loss_for_window_batch_tensor, temp_embedded_inputs
                    del temp_attention_mask, logits_last, next_token_logits_last, next_token_log_probs_last
                    del true_log_probs_for_token, true_ids_for_token, loss_for_token_step, avg_consistency_loss_for_batch_window_tensor
                    del true_next_token_embeddings, attention_for_new_token, output, fluency_loss_for_batch
                
                best_candidate_idx = np.argmin(candidate_losses)
                deobf_sys_prompt_ids = candidates[best_candidate_idx]

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

            deobf_sys_prompt_str = model_wrapper.tokenizer.decode(deobf_sys_prompt_ids)
            logger.info(f"Current obfuscated system prompt: {deobf_sys_prompt_str}")

            logger.info(f"Max GPU Utilization: {np.max(gpu_memory_used)//1024**2} MB")
            deobf_sys_prompt_list.append(deobf_sys_prompt_ids)

        # Append the true token IDs for the current window to the cumulative true token IDs
        true_ids_this_window_transposed = current_window_target_ids_full.transpose(0, 1)

        cumulative_true_ids_for_dataset = torch.cat(
            [cumulative_true_ids_for_dataset, true_ids_this_window_transposed], dim=1
        )
        cumulative_tokens_offset += num_tokens_in_window
        logger.info(f"Finished token window {token_window_idx + 1}. Cumulative true tokens appended: {num_tokens_in_window}. Total cumulative: {cumulative_tokens_offset}")
        del current_window_target_probs_full, current_window_target_ids_full, true_ids_this_window_transposed

    logger.info("Hard prompt deobfuscation finished.")
            
    return deobf_sys_prompt_list, history_train_loss_per_iteration
                    




def main(
    results_dir: str,
    embedding_file: str,
    deobfuscation_method: str,
    batch_size: int,
    dataset_size: int,
    output_token_count: int,
    window_size: int,
    optimizer_iter: int,
    lr: float,
    topk: int,
    search_width: int,
    n_replace: int,
    ce_weight: float,
    kl_weight: float,
    consistency_loss_weight: float,
    fluency_loss_weight: float,
    deobfuscated_sys_prompts_filename: str,
) -> None:
    """
    Main function to run the fluency deobfuscation attack on an obfuscated soft prompt.

    1.  Loads the configuration from the original obfuscation run.
    2.  Loads unseen data to be used for the attack, preventing data leakage.
    3.  Loads the target obfuscated soft prompt embedding.
    4.  Precomputes the "ground truth" outputs generated by this obfuscated embedding.
        These outputs will be the target for the deobfuscation optimization.
    5.  Runs the chosen deobfuscation method ('soft' or 'hard') to find a new,
        more readable prompt that mimics the behavior of the obfuscated one.
    6.  Saves the list of deobfuscated prompt candidates generated during the process.

    Args:
        results_dir (str): Path to the directory where obfuscate.py saved its results.
        embedding_file (str): Path to the tensor file containing embeddings.
        deobfuscation_method (str): Method for deobfuscating the embedded system prompt.
        batch_size (int): Batch size for optimization.
        dataset_size (int): Dataset size for optimization.
        output_token_count (int): Number of output tokens to optimize over.
        window_size (int): Number of tokens in the context window for gradient calculation.
        optimizer_iter (int): Number of optimization iterations.
        lr (float): Learning rate for optimization (soft prompt deobfuscation).
        topk (int): topk value for GCG (hard prompt deobfuscation).
        search_width (int): search_width value for GCG (hard prompt deobfuscation).
        n_replace (int): n_replace value for GCG (hard prompt deobfuscation).
        ce_weight (float): Weight for cross-entropy loss.
        kl_weight (float): Weight for KL divergence loss.
        consistency_loss_weight (float): Weight for consistency loss.
        fluency_loss_weight (float): Weight for fluency loss.
        deobfuscated_sys_prompts_filename (str): Filename for the output file containing deobfuscated system prompts.
    """
    logger = logging.getLogger(__name__)
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    params_file = results_dir / "params.json"
    if not params_file.exists():
        logger.error(f"params.json not found in {results_dir}")
        sys.exit(1)
    with open(params_file, "r") as f:
        params = json.load(f)
    logger.info(f"Loaded obfuscation parameters: {json.dumps(params, indent=2)}")

    seed = params["seed"]
    set_seed(seed)

    quantization_mode = None
    if params.get("quantize_4bit", False):
        quantization_mode = "4bit"
    elif params.get("quantize_8bit", False):
        quantization_mode = "8bit"
    
    try:
        model_wrapper = Model(params["model_name"], quantization_mode)
    except Exception as e:
        logger.exception(f"Failed to load model '{params['model_name']}'. Error: {e}")
        return
    logger.info(f"Loaded model: {params['model_name']}")

    original_dataset_name = params['dataset_name']
    original_dataset_size = params['dataset_size']
    task_hints = params['task_hints']
    logger.info("Loading dataset...")

    # Deobfuscation, we want different data than obfuscation. Since truthfulqa does not contain enough samples
    # choose triviaqa for that case. In the other cases, choose the same dataset, but different samples.
    if original_dataset_name.lower() == 'truthfulqa':
        try:
            train_samples, _, task_system_prompt = load_and_prepare_dataset(
                'triviaqa',
                dataset_size,
                task_hints,
                seed,
                split_ratio=0.8
            )
        except ValueError as e:
            logger.error(f"Error during dataset preparation: {e}")
            return
        except Exception as e:
            logger.exception(f"An unexpected error occurred during dataset preparation for triviaqa.")
            return
    else:
        try:
            train_samples, _, task_system_prompt = load_and_prepare_dataset(
                original_dataset_name,
                dataset_size+original_dataset_size,
                task_hints,
                seed,
                split_ratio=0.8
            )
        except ValueError as e:
            logger.error(f"Error during dataset preparation: {e}")
            return
        except Exception as e:
            logger.exception(f"An unexpected error occurred during dataset preparation for {original_dataset_name}.")
            return

    if original_dataset_name.lower() != 'truthfulqa':
        train_samples = list(np.asarray(train_samples)[original_dataset_size:])
    
    logger.debug(f"Example training sample: {train_samples[0]}")

    embedding_tensor = torch.load(embedding_file, weights_only=True)
    if embedding_tensor.dim() != 2:
        logger.error(f"Invalid embedding tensor shape: {embedding_tensor.shape}. Expected 2D tensor.")
        return

    pad_token_string = model_wrapper.tokenizer.pad_token
    # We use the outputs of the obfuscated system prompts as reference outputs for optimization
    placeholder_sys_prompt = f"{pad_token_string}Placeholder{pad_token_string}"
    placeholder_sys_ids = model_wrapper.tokenizer(
        placeholder_sys_prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0]

    train_dataset = TextDataset(train_samples)

    collate_fn = create_collate_fn(
        tokenizer=model_wrapper.tokenizer,
        system_prompt=placeholder_sys_prompt
    )
    placeholder_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    logger.info("Precomputing model outputs (probs and IDs) using obfuscated system prompt...")
    precomputed_probs, precomputed_ids, max_generated_length = precompute_model_outputs_replace(
        model_wrapper=model_wrapper,
        dataloader=placeholder_dataloader,
        max_new_tokens=output_token_count,
        sys_prompt_obf=embedding_tensor,
        original_sys_prompt_ids=placeholder_sys_ids,
        is_soft_prompt_method=True,
        obf_sys_prompt_len=embedding_tensor.shape[0],
        pad_token_id=model_wrapper.tokenizer.pad_token_id
    )
        
    output_token_count = max_generated_length
    if window_size >= output_token_count:
        window_size = output_token_count

    token_windows = create_non_overlapping_windows(output_token_count, window_size)
    logger.debug(f"Number of output token windows: {len(token_windows)}")

    if deobfuscation_method == 'soft':
        logger.info(f'Deobfuscating using soft prompt fluency optimization...')
        deobf_sys_prompt_list, train_loss_per_iteration = deobfuscate_soft_prompt(
            model_wrapper=model_wrapper,
            precomputed_probs=precomputed_probs,
            precomputed_ids=precomputed_ids,
            train_dataloader=placeholder_dataloader,
            sys_prompt_obf=embedding_tensor,
            original_sys_prompt_ids=placeholder_sys_ids,
            obf_sys_prompt_len=embedding_tensor.shape[0],
            lr=lr,
            optimizer_iter=optimizer_iter,
            token_windows=token_windows,
            ce_weight=ce_weight,
            kl_weight=kl_weight,
            consistency_loss_weight=consistency_loss_weight,
            fluency_loss_weight=fluency_loss_weight
        )
    elif deobfuscation_method == 'hard':
        logger.info(f'Deobfuscating using hard prompt fluency optimization...')
        deobf_sys_prompt_list, train_loss_per_iteration = deobfuscate_hard_prompt(
            model_wrapper=model_wrapper,
            precomputed_probs=precomputed_probs,
            precomputed_ids=precomputed_ids,
            train_dataloader=placeholder_dataloader,
            sys_prompt_obf=embedding_tensor,
            original_sys_prompt_ids=placeholder_sys_ids,
            obf_sys_prompt_len=embedding_tensor.shape[0],
            optimizer_iter=optimizer_iter,
            token_windows=token_windows,
            topk=topk,
            search_width=search_width,
            n_replace=n_replace,
            ce_weight=ce_weight,
            kl_weight=kl_weight,
            consistency_loss_weight=consistency_loss_weight,
            fluency_loss_weight=fluency_loss_weight
        )

    torch.save(deobf_sys_prompt_list, results_dir / deobfuscated_sys_prompts_filename)
    logger.info(f"Deobfuscated system prompts saved to {results_dir / deobfuscated_sys_prompts_filename}")

if __name__ == "__main__":
    setup_logging('fluency_deobfuscation.log', 'INFO') # Change to 'DEBUG' for more verbose logging
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