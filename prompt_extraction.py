import json
import logging
import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.utils import TextDataset, create_collate_fn
from src.logging_config import setup_logging
from src.model import Model
from src.output_generation import generate_model_responses_replace
from src.utils import set_seed


def get_args() -> Namespace:
    """Parses command-line arguments for the prompt extraction attack script."""
    parser = ArgumentParser(
        description="Script for running prompt extraction attack.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where obfuscate.py saved its results."
    )
    parser.add_argument(
        "--extraction_prompts_file",
        type=str,
        required=True,
        help="Path to the file containing extraction prompts."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generating model outputs during attack."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="prompt_extraction_output.json",
        help="Filename for the output file."
    )
    system_prompt_group = parser.add_mutually_exclusive_group(required=True)
    system_prompt_group.add_argument(
        "--conventional",
        action="store_true",
        help="Use the conventional system prompt in params.json."
    )
    system_prompt_group.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Specify a custom system prompt directly as a string."
    )
    system_prompt_group.add_argument(
        "--tensor_file",
        type=str,
        default=None,
        help="Specify a tensor file to load the system prompt from."
    )
    system_prompt_group.add_argument(
        "--blank",
        action="store_true",
        help="Use a blank system prompt."
    )
    args = parser.parse_args()
    
    return args

def get_sys_prompt(
    params: dict,
    conventional_sys_prompt: bool | None,
    system_prompt: str | None,
    tensor_file: str | None,
    blank: bool | None,
    pad_token_str: str,
) -> str | torch.Tensor | None:
    """
    Selects the appropriate system prompt based on command-line flags.

    Args:
        params (dict): Dictionary of parameters from a previous run.
        conventional_sys_prompt (bool | None): Flag to use the conventional system prompt.
        system_prompt (str | None): A custom system prompt string.
        tensor_file (str | None): Path to a tensor file containing the prompt.
        blank (bool | None): Flag to use a blank system prompt.
        pad_token_str (str): The string representation of the pad token.

    Returns:
        (str | torch.Tensor | None): The selected system prompt as a string or tensor.
    """
    sys_prompt = None
    if conventional_sys_prompt:
        logger.info("Using conventional system prompt.")
        sys_prompt = params["system_prompt"]
        logger.info(f"Conventional system prompt: {sys_prompt}")
    elif system_prompt:
        logger.info("Using custom system prompt.")
        sys_prompt = f"{pad_token_str}{system_prompt}{pad_token_str}"
        logger.info(f"Custom system prompt: {sys_prompt}")
    elif tensor_file:
        logger.info(f"Loading system prompt from tensor file {tensor_file}.")
        sys_prompt = torch.load(tensor_file, weights_only=True)
    elif blank:
        logger.info("Using blank system prompt.")
        sys_prompt = f"{pad_token_str}{pad_token_str}"
    
    return sys_prompt



def main(
    results_dir: str,
    extraction_prompts_file: str,
    batch_size: int,
    output_filename: str,
    conventional: bool | None,
    system_prompt: str | None,
    tensor_file: str | None,
    blank: bool | None,
) -> None:
    """
    Runs a prompt extraction attack against a model with a given system prompt.

    The script loads a model and a set of attack prompts. It configures the model
    with a specified system prompt (e.g., the original, an obfuscated one, or blank)
    and generates responses to the attack prompts. The results are saved for later
    evaluation by `evaluate_prompt_extraction.py`.

    Args:
        results_dir (str): Path to a directory containing configuration files.
        extraction_prompts_file (str): Path to the JSON file with extraction attack prompts.
        batch_size (int): Batch size for generating model responses.
        output_filename (str): Filename for saving the generated attack outputs.
        conventional (bool | None): If True, use the conventional system prompt.
        system_prompt (str | None): A custom system prompt string.
        tensor_file (str | None): Path to a tensor file containing the system prompt.
        blank (bool | None): If True, use a blank system prompt.
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

    set_seed(params.get("seed", 42))

    generation_config_file = results_dir / "generation_config.json"
    if not generation_config_file.exists():
        logger.error(f"generation_config.json not found in {results_dir}")
        sys.exit(1)
    with open(generation_config_file, "r") as f:
        generation_config = json.load(f)
    logger.info(f"Loaded generation config: {json.dumps(generation_config, indent=2)}")

    extraction_prompts = []
    with open(extraction_prompts_file, "r") as f:
        extraction_prompts = json.load(f)
    logger.info(f"Loaded {len(extraction_prompts)} extraction prompts.")

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

    pad_token_string = model_wrapper.tokenizer.pad_token
    sys_prompt = get_sys_prompt(
        params,
        conventional, 
        system_prompt, 
        tensor_file,
        blank,
        pad_token_string
    )

    if sys_prompt is None:
        logger.error("System prompt not found.")
        return

    # If sys prompt is str tokenize it
    if isinstance(sys_prompt, str):
        sys_prompt = model_wrapper.tokenizer(
            sys_prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids[0]
    
    # If system prompt is 1D --> Assume token IDs
    # If system prompt is 2D --> Assume embeddings
    if sys_prompt.dim() == 1:
        embedded = False
    elif sys_prompt.dim() == 2:
        embedded = True
    else:
        logger.error("System prompt has invalid dimensions. Must be 1D or 2D.")
        return

    placeholder_sys_prompt = f"{pad_token_string}Placeholder{pad_token_string}"
    placeholder_sys_ids = model_wrapper.tokenizer(
        placeholder_sys_prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0]
    extraction_dataset = TextDataset(extraction_prompts)
    collate_fn = create_collate_fn(
        tokenizer=model_wrapper.tokenizer,
        system_prompt=placeholder_sys_prompt,
    )
    placeholder_dataloader = DataLoader(
        extraction_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    extraction_output = generate_model_responses_replace(
        model_wrapper=model_wrapper,
        dataloader=placeholder_dataloader,
        generation_args=generation_config,
        sys_prompt_obf=sys_prompt,
        original_sys_prompt_ids=placeholder_sys_ids,
        is_soft_prompt_method=embedded,
        obf_sys_prompt_len=sys_prompt.shape[0],
        pad_token_id=model_wrapper.tokenizer.pad_token_id
    )

    output_dict = {
        "input": extraction_prompts,
        "output": extraction_output,
        "generation_config": generation_config,
        "conventional": conventional,
        "system_prompt": system_prompt,
        "tensor_file": tensor_file,
        "blank": blank,
        "seed": params["seed"]
    }

    with open(results_dir / output_filename, "w") as f:
        json.dump(output_dict, f, indent=4)

    logger.info(f"Saved extraction output to {results_dir / output_filename}.")


if __name__ == "__main__":
    setup_logging('prompt_extraction.log', 'INFO') # Change to 'DEBUG' for more verbose logging
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