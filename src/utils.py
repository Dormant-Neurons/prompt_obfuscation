import os
import random
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from tqdm import tqdm
from transformers import set_seed as huggingface_set_seed


def print_gpu_utilization() -> None:
    """Prints the current GPU memory usage to the console."""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    tqdm.write(f"GPU memory occupied: {info.used//1024**2} MB.")

def get_gpu_utilization() -> int:
    """Gets the memory utilization of the current GPU device in bytes."""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used


def set_seed(seed: int | None = None) -> None:
    """
    Sets all random seeds for reproducibility (deterministic mode).
    When seed is None, this function has no effect.

    Args:
        seed (int | None): The seed to set.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        huggingface_set_seed(seed)


def create_non_overlapping_windows(total_amount: int, window_size: int) -> list[list[int]]:
    """
    Creates a list of non-overlapping index windows.

    Args:
        total_amount (int): The total number of items to be windowed.
        window_size (int): The size of each window.

    Returns:
        (list[list[int]]): A list of lists, where each inner list contains the indices for a window.
    """
    if(window_size <= 0):
        raise ValueError("Window size must be a positive integer.")
    if(total_amount < 0):
        raise ValueError("Total amount must be a non-negative integer.")

    num_windows = ceil(total_amount / window_size)
    windows = []
    for i in range(num_windows):
        start_index = i * window_size
        end_index = min(start_index + window_size, total_amount)
        window = list(range(start_index, end_index))
        windows.append(window)

    return windows


def loss_function_with_padding_mask(
    pred_logits: torch.Tensor,
    pred_log_probs: torch.Tensor,
    true_log_probs: torch.Tensor,
    true_ids: torch.Tensor,         
    kl_weight: float,
    ce_weight: float,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Calculates a combined Cross-Entropy and KL Divergence loss,
    ignoring positions where the true token ID is the padding token.

    Args:
        pred_logits (torch.Tensor): Predicted logits from the model.
        pred_log_probs (torch.Tensor): Log probabilities from the model's prediction.
        true_log_probs (torch.Tensor): Target log probabilities (precomputed).
        true_ids (torch.Tensor): Target token IDs (precomputed).
        kl_weight (float): Weight for the KL divergence loss component.
        ce_weight (float): Weight for the Cross-Entropy loss component.
        pad_token_id (int): ID of the padding token to be ignored in the loss calculation.

    Returns:
        (torch.Tensor): The combined scalar loss value.
    """
    non_pad_mask = (true_ids != pad_token_id)

    if not non_pad_mask.any():
        return torch.tensor(0.0, dtype=pred_logits.dtype)
    
    
    total_loss_per_item = torch.zeros(pred_logits.size(0), dtype=pred_logits.dtype)

    if ce_weight > 0.0:
        ce_loss_per_item = F.cross_entropy(pred_logits, true_ids, reduction='none')
        total_loss_per_item += ce_weight * ce_loss_per_item

    if kl_weight > 0:
        kl_loss_fn_elementwise = torch.nn.KLDivLoss(reduction='none', log_target=True)
        elementwise_kl_div = kl_loss_fn_elementwise(pred_log_probs, true_log_probs)
        kl_loss_per_item = elementwise_kl_div.sum(dim=-1)
        total_loss_per_item += kl_weight * kl_loss_per_item
    
    masked_total_loss = total_loss_per_item[non_pad_mask]
    final_loss = masked_total_loss.mean()

    return final_loss


def find_best_candidate_by_rank(
    candidate_scores: list[dict[str, float]],
    metric_list: list[str],
    higher_is_better_map: dict[str, bool],
) -> tuple[int, dict[str, float]]:
    """
    Finds the best candidate from a list of scores by summing their ranks across specified metrics.

    For each specified metric, candidates are ranked. The ranks are then summed for each
    candidate, and the candidate with the lowest sum of ranks is considered the best.

    Args:
        candidate_scores (list[dict[str, float]]): A list of dictionaries, where each dict holds scores for a candidate.
        metric_list (list[str]): The list of metric names to consider for ranking.
        higher_is_better_map (dict[str, bool]): A map indicating if a higher score is better for each metric.

    Returns:
        (tuple[int, dict[str, float]]): A tuple containing:
            - The index of the best candidate in the original list.
            - A dictionary with the scores of the best candidate.
    """
    if not candidate_scores:
        raise ValueError("candidate_scores list is empty.")
    if not metric_list:
        raise ValueError("metric_list is empty.")

    df = pd.DataFrame(candidate_scores)
    
    rank_metrics = [m for m in metric_list if m in df.columns]
    if not rank_metrics:
        # Fallback to returning the first candidate if no metrics match
        return 0, df.iloc[0].to_dict()

    rank_df = pd.DataFrame(index=df.index)

    for metric in rank_metrics:
        # Determine sort order for ranking
        ascending = not higher_is_better_map.get(metric, True) # Default to True if not specified
        rank_df[metric + '_rank'] = df[metric].rank(method='min', ascending=ascending, na_option='bottom')
    
    df['rank_sum'] = rank_df.filter(like='_rank').sum(axis=1)
    
    best_idx = df['rank_sum'].idxmin()
    best_scores_dict = df.loc[best_idx, metric_list].to_dict()
    
    return int(best_idx), best_scores_dict