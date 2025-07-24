import logging
from itertools import combinations, product
from math import comb

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# Dictionary mapping model names to tuples of beginning and end token IDs for system prompts. We need to hardcode this, since this has to work with blank system prompts as well
MODEL_TEMPLATE_SYS_PROMPT_BEGINNING_END_IDS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": (271, 128009),
    "meta-llama/Llama-2-7b-chat-hf": (13, 13),
    "tiiuae/Falcon3-7B-Instruct": (12, 12)
}

def apply_chat_template_to_batch(
    user_prompts: list[str],
    system_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """
    Applies the tokenizer's chat template to a batch of user prompts
    with a given system prompt.

    Args:
        user_prompts (list[str]): A list of user input strings.
        system_prompt (str): The content for the system role.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.

    Returns:
        (list[str]): A list of formatted strings ready for tokenization.
    """
    formatted_prompts = []
    for user_prompt in user_prompts:
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({'role': 'user', 'content': user_prompt})
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        )
    return formatted_prompts


def zero_pad_token_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Zeros out the attention mask for all occurrences of the pad_token_id."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError(f"input_ids shape {input_ids.shape} and attention_mask shape {attention_mask.shape} must match.")
    
    new_attention_mask = attention_mask.clone()
    pad_token_positions = (input_ids == pad_token_id)
    new_attention_mask[pad_token_positions] = 0
    return new_attention_mask


def generate_random_token_sequence(
    num_tokens: int,
    vocab_size: int,
) -> torch.Tensor:
    """Generates a 1D tensor of random token IDs."""
    return torch.randint(0, vocab_size, (num_tokens,))


def _extract_delimited_prompt_segment(
    prompt_with_delimiters: torch.Tensor,
    delimiter_token_id: int,
) -> torch.Tensor:
    """
    Extracts the segment of a prompt that is enclosed by delimiter tokens,
    including the delimiter tokens themselves. It expects exactly two delimiter tokens.

    Args:
        prompt_with_delimiters (torch.Tensor): 1D Tensor containing the prompt IDs.
        delimiter_token_id (int): The ID of the delimiter token.

    Returns:
        (torch.Tensor): The 1D segment of the prompt including the two delimiter tokens.
    """
    if not isinstance(prompt_with_delimiters, torch.Tensor):
        raise TypeError("prompt_with_delimiters must be a torch.Tensor.")
    if not prompt_with_delimiters.ndim == 1:
        raise ValueError("prompt_with_delimiters must be a 1D tensor.")

    delimiter_positions = torch.nonzero(prompt_with_delimiters == delimiter_token_id, as_tuple=False).squeeze(-1)

    if delimiter_positions.numel() != 2:
        raise ValueError(
            f"Input tensor for _extract_delimited_prompt_segment must contain exactly two "
            f"delimiter_token_id ({delimiter_token_id}) tokens. Found {delimiter_positions.numel()} "
            f"at positions {delimiter_positions.tolist()} in tensor of size {prompt_with_delimiters.size(0)}."
        )

    start_idx = delimiter_positions[0].item()
    end_idx = delimiter_positions[1].item()

    return prompt_with_delimiters[start_idx : end_idx + 1]

def find_sys_prompt_indices(
    input_ids: torch.Tensor,
    sys_prompt_ids: torch.Tensor,
    pad_token_id: int,
    model_name: str,
) -> tuple[int, int]:
    """
    Finds the start and end indices of the system prompt content within a fully templated input.

    Args:
        input_ids (torch.Tensor): A 1D tensor of token IDs for the full, templated input.
        sys_prompt_ids (torch.Tensor): A 1D tensor of token IDs encoding the system prompt content.
        pad_token_id (int): The ID of the padding token used as a delimiter.
        model_name (str): The name of the model, used to look up template tokens.

    Returns:
        (tuple[int, int]): A tuple containing the start and end indices of the system prompt content.
    """
    sys_prompt_ids = _extract_delimited_prompt_segment(sys_prompt_ids, pad_token_id)

    template_beginning_id, template_end_id = MODEL_TEMPLATE_SYS_PROMPT_BEGINNING_END_IDS[model_name]
    first_id_tensor = torch.tensor([template_beginning_id], dtype=sys_prompt_ids.dtype, device=sys_prompt_ids.device)
    second_id_tensor = torch.tensor([template_end_id], dtype=sys_prompt_ids.dtype, device=sys_prompt_ids.device)
    sys_prompt_ids = torch.cat([first_id_tensor, sys_prompt_ids, second_id_tensor])
    
    input_len = input_ids.size(0)
    sys_prompt_len = sys_prompt_ids.size(0)

    for i in range(input_len - sys_prompt_len + 1):
        sub_tensor = input_ids[i:i + sys_prompt_len]
        if(torch.equal(sub_tensor, sys_prompt_ids)):
            start_index = i
            end_index = i + sys_prompt_len - 1
            return (start_index+2, end_index-1)

    raise ValueError("System prompt not found in input tensor.")


def find_sys_prompt_indices_batch(
    input_ids: torch.Tensor,
    sys_prompt_ids: torch.Tensor,
    pad_token_id: int,
    model_name: str,
) -> list[tuple[int, int]]:
    """Applies `find_sys_prompt_indices` to a batch of inputs."""
    indices = [find_sys_prompt_indices(input_id, sys_prompt_ids, pad_token_id, model_name) for input_id in input_ids]
    return indices


def update_attention_mask(
    obf_sys_prompt_len: int,
    attention_mask: torch.Tensor,
    sys_prompt_indices: tuple[int, int],
) -> torch.Tensor:
    """Updates an attention mask to account for a replaced system prompt of a different length."""
    start_index, end_index = sys_prompt_indices
    return torch.cat((attention_mask[:start_index], torch.ones(obf_sys_prompt_len), attention_mask[end_index:]), dim=0)

def update_attention_mask_batch(
    obf_sys_prompt_len: int,
    attention_masks: torch.Tensor,
    sys_prompt_indices: list[tuple[int, int]],
) -> torch.Tensor:
    """Applies `update_attention_mask` to a batch of attention masks."""
    new_attention_masks = [update_attention_mask(obf_sys_prompt_len, attention_mask, sys_prompt_indices[idx]) for idx, attention_mask in enumerate(attention_masks)]
    return torch.stack(new_attention_masks, dim=0)


def replace_sys_prompt(
    new_sys_prompt: torch.Tensor,
    prompt: torch.Tensor,
    sys_prompt_indices: tuple[int, int],
) -> torch.Tensor:
    """Replaces the system prompt segment in a tensor with a new system prompt."""
    start_index, end_index = sys_prompt_indices
    return torch.cat((prompt[:start_index], new_sys_prompt, prompt[end_index:]), dim=0)
    

def replace_sys_prompt_batch(
    new_sys_prompt: torch.Tensor,
    prompts: torch.Tensor,
    sys_prompt_indices: list[tuple[int, int]],
) -> torch.Tensor:
    """Applies `replace_sys_prompt` to a batch of prompts."""
    new_prompts = [replace_sys_prompt(new_sys_prompt, prompt, sys_prompt_indices[idx]) for idx, prompt in enumerate(prompts)]
    return torch.stack(new_prompts, dim=0)


def _calculate_max_candidates(ids_length: int, topk: int, n_replace: int) -> int:
    """Calculates the maximum number of unique candidate sequences for GCG."""
    if n_replace > ids_length:
        return 0
    position_combinations = comb(ids_length, n_replace)
    max_candidates = float(position_combinations) * (topk ** n_replace)
    return int(max_candidates) if max_candidates < float('inf') else float('inf')

def _get_all_possible_candidates(
    ids: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    n_replace: int
) -> torch.Tensor:
    """Generates all possible GCG candidates when search space is small."""
    ids_length = len(ids)
    position_combinations = list(combinations(range(ids_length), n_replace))
    token_combinations = list(product(range(topk), repeat=n_replace))

    all_candidates = []
    for pos_comb in position_combinations:
        for token_comb in token_combinations:
            new_candidate = ids.clone()
            for pos, token_index in zip(pos_comb, token_comb):
                new_candidate[pos] = topk_ids[pos, token_index]
            all_candidates.append(new_candidate)

    return torch.stack(all_candidates, dim=0)

def _sample_candidates(
    ids: torch.Tensor,
    topk_ids: torch.Tensor,
    search_width: int,
    topk: int,
    n_replace: int,
) -> torch.Tensor:
    """
    Samples a set of GCG candidate sequences.
    Inspired by https://github.com/GraySwanAI/nanoGCG/blob/main/nanogcg/gcg.py
    """
    ids_length = len(ids)
    original_ids = ids.repeat(search_width, 1)

    rand_matrix = torch.rand(search_width, ids_length)
    positions_to_replace = rand_matrix.argsort(dim=1)[:, :n_replace]

    topk_for_positions = topk_ids[positions_to_replace]
    rand_topk_indices = torch.randint(0, topk, (search_width, n_replace, 1))
    replacement_values = torch.gather(topk_for_positions, 2, rand_topk_indices).squeeze(-1)

    return original_ids.scatter(1, positions_to_replace, replacement_values)


def get_GCG_candidates(
    ids: torch.Tensor,
    grad: torch.Tensor,
    search_width: int,
    topk: int,
    n_replace: int,
) -> torch.Tensor:
    """
    Generates candidate token sequences using the Greedy Coordinate Gradient (GCG) method.

    This function first identifies the `topk` best token substitutions for each position based
    on the input gradient. It then either exhaustively generates all possible combinations of
    `n_replace` substitutions or samples `search_width` candidates if the total number of
    combinations is too large.

    NOTE: This function may return a tensor with fewer rows than `search_width` if the
    total number of unique candidates is less than `search_width`. The calling code
    must handle a variable number of candidates.

    Args:
        ids (torch.Tensor): The current token IDs.
        grad (torch.Tensor): The gradient of the loss with respect to the one-hot encoded tokens.
        search_width (int): The number of candidates to generate.
        topk (int): The number of top token replacements to consider for each position.
        n_replace (int): The number of tokens to replace in each candidate.

    Returns:
        (torch.Tensor): A tensor of candidate token sequences.
    """
    ids_length = len(ids)
    # Get the top-k token IDs with the highest gradients for each position
    topk_ids = (-grad).topk(topk, dim=1).indices.cpu()
    max_candidates = _calculate_max_candidates(ids_length, topk, n_replace)
    
    #Heuristic: If the search space is not significantly larger than the number of candidates we need,
    #it's more robust and efficient to generate them all. A factor of 1.5 is a good balance between performance and sampling quality
    if max_candidates <= search_width * 1.5:
        all_candidates = _get_all_possible_candidates(ids, topk_ids, topk, n_replace)

        num_generated = all_candidates.shape[0]
        if num_generated == 0:
            return ids.unsqueeze(0)
        
        shuffled_indices = torch.randperm(num_generated)
        num_to_return = min(num_generated, search_width)
        return all_candidates[shuffled_indices[:num_to_return]]
    else:
        return _sample_candidates(ids, topk_ids, search_width, topk, n_replace)