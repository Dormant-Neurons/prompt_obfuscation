import logging

import numpy as np
import textdistance
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction

logger = logging.getLogger(__name__)

def levenshtein(sys_prompt_1: str, sys_prompt_2: str) -> float:
    """Computes the normalized Levenshtein similarity between two strings."""
    distance = textdistance.levenshtein.distance(sys_prompt_1, sys_prompt_2)
    max_len = max(len(sys_prompt_1), len(sys_prompt_2))
    if max_len == 0:
        return 1.0
    normalized_similarity = 1 - (distance / max_len)
    return normalized_similarity


def jaccard(sys_prompt_1: str, sys_prompt_2: str) -> float:
    """Computes the Jaccard similarity based on character n-grams."""
    n=3
    def get_ngrams(s: str, n: int) -> set:
        return set([s[i:i+n] for i in range(len(s) - n + 1)])

    ngrams1 = get_ngrams(sys_prompt_1, n)
    ngrams2 = get_ngrams(sys_prompt_2, n)

    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    if(not union):
        return 1.0

    jaccard_index = len(intersection) / len(union)

    return jaccard_index

def lcs(sys_prompt_1: str, sys_prompt_2: str) -> float:
    """Computes the normalized Longest Common Subsequence (LCS) similarity."""
    lcs_length = textdistance.lcsseq.similarity(sys_prompt_1, sys_prompt_2)
    
    max_len = max(len(sys_prompt_1), len(sys_prompt_2))
    
    if(max_len == 0):
        return 1.0
    
    normalized_similarity = lcs_length / max_len
    
    return normalized_similarity

def cosine_similarity(sys_prompt_1: str, sys_prompt_2: str) -> float:
    """Computes the cosine similarity of two strings using sentence-transformer embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-mpnet-base-v2", trust_remote_code=True, device=device)
    model.similarity_fn_name = SimilarityFunction.COSINE
    try:
        sys_prompt_1_embeddings = model.encode(sys_prompt_1, convert_to_tensor=True)
        sys_prompt_2_embeddings = model.encode(sys_prompt_2, convert_to_tensor=True)
        similarity = model.similarity(sys_prompt_1_embeddings, sys_prompt_2_embeddings).cpu().numpy().item()
    except Exception as e:
        logger.warning(f"Error in cosine similarity: {e}")
        similarity = np.nan
    return similarity



AVAILABLE_METRICS = {
    "levenshtein": levenshtein,
    "jaccard": jaccard,
    "lcs": lcs,
    "cosine_similarity": cosine_similarity
}

def compute_sys_prompt_similarity(
    sys_prompt_1: str,
    sys_prompt_2: str,
    metric_list: list[str],
) -> dict[str, float]:
    """
    Computes a set of similarity scores between two system prompt strings.

    Args:
        sys_prompt_1 (str): The first system prompt string.
        sys_prompt_2 (str): The second system prompt string.
        metric_list (list[str]): A list of metric names to compute.

    Returns:
        (dict[str, float]): A dictionary mapping metric names to their computed scores.
    """
    result_dir = {}
    for metric_name in metric_list:
        metric_func = AVAILABLE_METRICS.get(metric_name, None)
        if(metric_func is None):
            logger.warning(f"Metric {metric_name} is not available. Skipping.")
            continue

        score = metric_func(sys_prompt_1.lower(), sys_prompt_2.lower())
        result_dir[metric_name] = score
    return result_dir