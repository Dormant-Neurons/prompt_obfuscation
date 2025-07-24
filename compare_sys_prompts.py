import json
import logging
import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.logging_config import setup_logging
from src.prompt_utils import generate_random_token_sequence
from src.sys_prompt_similarity import compute_sys_prompt_similarity
from src.utils import set_seed


def get_args() -> Namespace:
    """Parses and validates command-line arguments for the system prompt comparison script."""
    parser = ArgumentParser(
        description="Script for comparing system prompts.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where obfuscate.py saved its results."
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=["levenshtein", "jaccard", "lcs", "cosine_similarity"],
        default=["levenshtein", "jaccard", "lcs", "cosine_similarity"],
        help="List of metrics to use for evaluation."
    )
    sys_prompt_1_group = parser.add_mutually_exclusive_group(required=True)
    sys_prompt_1_group.add_argument(
        "--sys_prompt_1_conventional",
        action="store_true",
        help="Use the conventional system prompt from params.json."
    )
    sys_prompt_1_group.add_argument(
        "--sys_prompt_1_file",
        type=str,
        help="Path to tensor ID file containing the first system prompt."
    )
    sys_prompt_1_group.add_argument(
        "--sys_prompt_1_string",
        type=str,
        help="String containing the first system prompt."
    )
    sys_prompt_1_group.add_argument(
        "--sys_prompt_1_random",
        action="store_true",
        help="Generate a random system prompt."
    )
    sys_prompt_2_group = parser.add_mutually_exclusive_group(required=True)
    sys_prompt_2_group.add_argument(
        "--sys_prompt_2_conventional",
        action="store_true",
        help="Use the conventional system prompt from params.json."
    )
    sys_prompt_2_group.add_argument(
        "--sys_prompt_2_file",
        type=str,
        help="Path to tensor ID file containing the second system prompt."
    )
    sys_prompt_2_group.add_argument(
        "--sys_prompt_2_string",
        type=str,
        help="String containing the second system prompt."
    )
    sys_prompt_2_group.add_argument(
        "--sys_prompt_2_random",
        action="store_true",
        help="Generate a random system prompt."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the scores will be saved."
    )
    parser.add_argument(
        "--scores_filename",
        type=str,
        default="scores.json",
        help="Filename for the output score file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility."
    )
    args = parser.parse_args()

    return args

def load_sys_prompt(
    params: dict,
    conventional: bool | None,
    string: str | None,
    tensor_file: str | None,
    random: bool | None,
    sys_prompt_len: int,
    vocab_size: int,
) -> str | torch.Tensor | None:
    """
    Loads or generates a system prompt based on the provided arguments.

    Args:
        params (dict): Dictionary of parameters from params.json, used for the conventional prompt.
        conventional (bool | None): Flag to use the conventional system prompt from params.
        string (str | None): A string to be used directly as the system prompt.
        tensor_file (str | None): Path to a .pt file containing token IDs.
        random (bool | None): Flag to generate a random token sequence.
        sys_prompt_len (int): The length for the random prompt.
        vocab_size (int): The vocabulary size for the random prompt.

    Returns:
        (str | torch.Tensor | None): The loaded or generated system prompt, which can be
                                     a string, a tensor of token IDs, or None if no option is selected.
    """
    sys_prompt = None
    if conventional:
        logger.info("Using conventional system prompt.")
        sys_prompt = params["system_prompt"]
        logger.info(f"Conventional system prompt: {sys_prompt}")
    elif string:
        logger.info("Using custom system prompt.")
        sys_prompt = string
        logger.info(f"Custom system prompt: {sys_prompt}")
    elif tensor_file:
        logger.info(f"Loading system prompt from tensor file {tensor_file}.")
        sys_prompt = torch.load(tensor_file, weights_only=True)
    elif random:
        logger.info("Generating random system prompt.")
        sys_prompt = generate_random_token_sequence(sys_prompt_len, vocab_size)
        logger.info(f"Random system prompt: {sys_prompt}")
    
    return sys_prompt


def main(
    results_dir: str,
    metrics: list[str],
    sys_prompt_1_conventional: bool | None,
    sys_prompt_1_file: str | None,
    sys_prompt_1_string: str | None,
    sys_prompt_1_random: bool | None,
    sys_prompt_2_conventional: bool | None,
    sys_prompt_2_file: str | None,
    sys_prompt_2_string: str | None,
    sys_prompt_2_random: bool | None,
    output_dir: str,
    scores_filename: str,
    seed: int,
) -> None:
    """
    Loads two system prompts from various sources, computes their similarity, and saves the scores.

    Args:
        results_dir (str): Path to the directory containing `params.json`.
        metrics (list[str]): List of similarity metrics to compute.
        sys_prompt_1_conventional (bool | None): Flag to use the conventional prompt for prompt 1.
        sys_prompt_1_file (str | None): Path to a tensor file for prompt 1.
        sys_prompt_1_string (str | None): A string to use for prompt 1.
        sys_prompt_1_random (bool | None): Flag to generate a random prompt for prompt 1.
        sys_prompt_2_conventional (bool | None): Flag to use the conventional prompt for prompt 2.
        sys_prompt_2_file (str | None): Path to a tensor file for prompt 2.
        sys_prompt_2_string (str | None): A string to use for prompt 2.
        sys_prompt_2_random (bool | None): Flag to generate a random prompt for prompt 2.
        output_dir (str): The directory where the resulting scores will be saved.
        scores_filename (str): Filename for the output JSON file.
        seed (int): Seed for reproducibility.
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

    new_seed = seed if seed is not None else params["seed"]
    set_seed(new_seed)

    obf_sys_prompt_len = params["obf_sys_prompt_len"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
        logger.info(f"Loaded tokenizer for: {params['model_name']}")
    except Exception as e:
        logger.exception(f"Failed to load tokenizer for '{params['model_name']}'. Error: {e}")
        return
    
    new_pad_token = "<|pad|>"
    tokenizer.add_special_tokens({"pad_token": new_pad_token})
    vocab_size = len(tokenizer)
    pad_token_string = tokenizer.pad_token

    sys_prompt_1 = load_sys_prompt(
        params,
        sys_prompt_1_conventional,
        sys_prompt_1_string,
        sys_prompt_1_file,
        sys_prompt_1_random,
        obf_sys_prompt_len,
        vocab_size
    )
    sys_prompt_2 = load_sys_prompt(
        params,
        sys_prompt_2_conventional,
        sys_prompt_2_string,
        sys_prompt_2_file,
        sys_prompt_2_random,
        obf_sys_prompt_len,
        vocab_size
    )

    if isinstance(sys_prompt_1, torch.Tensor):
        try:
            sys_prompt_1 = tokenizer.decode(sys_prompt_1, skip_special_tokens=False)
        except Exception as e:
            logger.warning(f"Error decoding tensor for sys prompt 1: {e}")
            sys.exit(1)
    if isinstance(sys_prompt_2, torch.Tensor):
        try:
            sys_prompt_2 = tokenizer.decode(sys_prompt_2, skip_special_tokens=False)
        except Exception as e:
            logger.warning(f"Error decoding tensor for sys prompt 2: {e}")
            sys.exit(1)

    #Remove pad token from sys prompts
    sys_prompt_1 = sys_prompt_1.replace(pad_token_string, "")
    sys_prompt_2 = sys_prompt_2.replace(pad_token_string, "")

    logger.info(f"System prompt 1: {sys_prompt_1}")
    logger.info(f"System prompt 2: {sys_prompt_2}")

    if sys_prompt_1 == sys_prompt_2:
        logger.info("System prompts are the same!")
        sys.exit(0)

    
    assert isinstance(sys_prompt_1, str) and isinstance(sys_prompt_2, str)

    logger.debug("Computing similarity...")
    scores = compute_sys_prompt_similarity(
        sys_prompt_1=sys_prompt_1,
        sys_prompt_2=sys_prompt_2,
        metric_list=metrics
    )

    logger.info(f"Scores: {scores}")

    scores_file = Path(output_dir) / scores_filename
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)
    
    logger.info(f"Saved scores to {scores_file}")

if __name__ == "__main__":
    setup_logging('compare_sys_prompts.log', 'INFO') # Change to 'DEBUG' for more verbose logging
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