import json
import logging
import sys

import torch
import nltk
nltk.download('perluniprops', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Configure logging to suppress verbose output from libraries
from src.logging_config import setup_logging

setup_logging('basic_test.log', 'INFO')
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('filelock').setLevel(logging.WARNING)
logging.getLogger('accelerate').setLevel(logging.WARNING)
logging.getLogger('bitsandbytes').setLevel(logging.WARNING)
logging.getLogger('nltk').setLevel(logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader

# Main components to test
from data.loader import load_and_prepare_dataset
from data.utils import TextDataset, create_collate_fn
from src.model import Model
from src.output_generation import generate_model_responses
from src.output_similarity import AVAILABLE_METRICS as OUTPUT_METRICS
from src.output_similarity import (DERIVED_METRICS_SOURCES,
                                   compute_similarity_scores)
from src.sys_prompt_similarity import AVAILABLE_METRICS as SYS_PROMPT_METRICS
from src.sys_prompt_similarity import compute_sys_prompt_similarity

console = Console()


def run_test_step(title, func):
    """Helper function to run a test step and print its status."""
    console.print(f"[bold yellow]Running: {title}...[/bold yellow]")
    try:
        result = func()
        console.print(Panel(f"[bold green]SUCCESS:[/] {title} completed.", expand=False, border_style="green"))
        return result
    except SystemExit as e:
        # Catch sys.exit to prevent a full traceback on controlled exits
        sys.exit(e.code)
    except Exception:
        console.print(Panel(f"[bold red]FAILED:[/] {title} encountered an error.", expand=False, border_style="red"))
        console.print_exception(show_locals=False)
        sys.exit(1)


def test_gpu_availability():
    """Checks for GPU availability and exits if not found."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        console.print(Panel(f"GPU detected: [bold green]{gpu_name}[/]", expand=False, border_style="green"))
    else:
        error_message = "[bold red]ERROR: No CUDA-enabled GPU detected.[/]"
        console.print(Panel(error_message, expand=False, border_style="red", title="[bold]Hardware Requirement Not Met[/bold]"))
        sys.exit(1)  # Exit with a non-zero error code


def test_model_loading():
    """Tests loading the model and tokenizer with quantization."""
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    console.print("This step requires Hugging Face login and access to Llama-3.1-8B.")
    console.print("It will download the model (if not cached), which can take time and space.")
    model_wrapper = Model(name_or_path=model_name, quantization_mode="4bit")
    assert model_wrapper.model is not None, "Model failed to load."
    assert model_wrapper.tokenizer is not None, "Tokenizer failed to load."
    console.print(f"Model '[cyan]{model_name}[/]' and its tokenizer loaded successfully.")
    return model_wrapper


def test_data_loading():
    """Tests loading and preparing a small slice of a dataset."""
    train_texts, test_texts, _ = load_and_prepare_dataset(
        dataset_name="truthfulqa",
        dataset_size=4,  # Use a very small size for a quick test
        use_task_hints=False,
        seed=42,
        split_ratio=0.5
    )
    assert len(train_texts) == 2, "Incorrect number of training samples."
    assert len(test_texts) == 2, "Incorrect number of test samples."
    console.print("Dataset '[cyan]truthfulqa[/]' loaded and processed successfully.")
    return train_texts


def test_output_generation(model_wrapper: Model, train_texts: list):
    """Tests the model's output generation capability."""
    system_prompt = "You are a helpful assistant."
    test_dataset = TextDataset(train_texts)
    collate_fn = create_collate_fn(tokenizer=model_wrapper.tokenizer, system_prompt=system_prompt)
    dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

    generation_config = {
        "max_new_tokens": 50,
        "num_return_sequences": 1,
        "temperature": 0.1,  # Low temp for more deterministic output
    }

    outputs = generate_model_responses(model_wrapper, dataloader, generation_config)

    assert isinstance(outputs, list) and len(outputs) > 0, "Output generation failed."
    assert isinstance(outputs[0], list) and isinstance(outputs[0][0], str), "Output format is incorrect."
    console.print("Generated Outputs (sample):")
    console.print(json.dumps(outputs, indent=2))
    console.print("Model response generation test successful.")


def test_output_similarity():
    """Tests the output similarity metric calculation for all available metrics."""
    references = [["The quick brown fox jumps over the lazy dog."]]
    predictions = [["The fast brown fox leaped over the lazy dog.", "a quick brown dog jumps over the lazy fox"]]
    
    # Get all metric keys, including derived ones
    all_metrics = list(OUTPUT_METRICS.keys()) + list(DERIVED_METRICS_SOURCES.keys())
    # Remove duplicates and the base 'rouge' key
    metric_list_to_test = sorted(list(set(all_metrics) - {'rouge'}))

    console.print(f"Testing {len(metric_list_to_test)} output similarity metrics...")
    scores = compute_similarity_scores(
        predictions=predictions,
        references=references,
        metric_list=metric_list_to_test
    )
    
    assert len(scores) == len(metric_list_to_test), "Failed to compute all similarity scores."
    console.print("Computed Output Similarity Scores:")
    console.print(json.dumps(scores, indent=2))
    for metric, score in scores.items():
        assert isinstance(score, (float, int)), f"Score for {metric} is not a number."
    console.print("Output similarity calculation test successful.")


def test_sys_prompt_similarity():
    """Tests the system prompt similarity metric calculation for all available metrics."""
    prompt1 = "Reply in the style of a pirate."
    prompt2 = "Answer me like a pirate, matey."
    
    metric_list_to_test = sorted(list(SYS_PROMPT_METRICS.keys()))

    console.print(f"Testing {len(metric_list_to_test)} system prompt similarity metrics...")
    scores = compute_sys_prompt_similarity(
        sys_prompt_1=prompt1,
        sys_prompt_2=prompt2,
        metric_list=metric_list_to_test
    )
    assert len(scores) == len(metric_list_to_test), "Failed to compute all sys prompt similarity scores."
    console.print("Computed System Prompt Similarity Scores:")
    console.print(json.dumps(scores, indent=2))
    for metric, score in scores.items():
        assert isinstance(score, float), f"Score for {metric} is not a float."
    console.print("System prompt similarity calculation test successful.")


def main():
    """Main function to run all basic tests."""
    console.rule("[bold magenta]Running Basic Functionality Test[/bold magenta]")
    console.print("This script will check that all main components of the project are functioning correctly.")
    
    # Run tests sequentially
    run_test_step("GPU Availability Check", test_gpu_availability)
    model_wrapper = run_test_step("Model and Tokenizer Loading", test_model_loading)
    train_data = run_test_step("Data Loading and Processing", test_data_loading)
    run_test_step(
        "Model Output Generation",
        lambda: test_output_generation(model_wrapper, train_data)
    )
    run_test_step("Output Similarity Calculation", test_output_similarity)
    run_test_step("System Prompt Similarity Calculation", test_sys_prompt_similarity)

    console.rule("[bold green]All Basic Tests Passed Successfully![/bold green]")
    console.print("The environment is set up correctly and all core components are functional.")


if __name__ == "__main__":
    main()