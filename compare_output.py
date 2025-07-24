import json
import logging
import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

from src.logging_config import setup_logging
from src.output_similarity import (AVAILABLE_METRICS, DERIVED_METRICS_SOURCES,
                                   HIGHER_IS_BETTER, compute_similarity_scores)
from src.utils import set_seed


def get_args() -> Namespace:
    """Parses and validates command-line arguments for the output comparison script."""
    parser = ArgumentParser(
        description="Script for comparing outputs.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--output_file_1",
        type=str,
        required=True,
        help="Path to the file containing reference outputs."
    )
    parser.add_argument(
        "--output_file_2",
        type=str,
        required=True,
        help="Path to the file containing candidate outputs."
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=list(HIGHER_IS_BETTER.keys()),
        default=["sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity"],
        help="List of metrics to use for evaluation."
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

    valid_metrics = list(AVAILABLE_METRICS.keys()) + list(DERIVED_METRICS_SOURCES.keys())
    for metric in args.metrics:
        if metric not in valid_metrics:
            parser.error(f"Invalid metric: {metric}. Choices are: {valid_metrics}")
    return args
    

def main(
    output_file_1: str,
    output_file_2: str,
    metrics: list[str],
    output_dir: str,
    scores_filename: str,
    seed: int,
) -> None:
    """
    Loads reference and candidate outputs from files, computes similarity scores, and saves them.

    Args:
        output_file_1 (str): Path to the JSON file with reference outputs.
        output_file_2 (str): Path to the JSON file with candidate outputs to be evaluated.
        metrics (list[str]): A list of metric names to compute.
        output_dir (str): The directory where the results will be saved.
        scores_filename (str): The name of the file to save the final scores in.
        seed (int): An integer seed for reproducibility of certain metrics.
    """
    logger = logging.getLogger(__name__)

    with open(output_file_1, "r") as f:
        ref_outputs: dict = json.load(f)
    with open(output_file_2, "r") as f:
        cand_outputs: dict = json.load(f)

    set_seed(seed)

    logger.info(f"Computing similarity scores for metrics: {metrics}")
    scores = compute_similarity_scores(
        predictions=cand_outputs["output"],
        references=ref_outputs["output"],
        metric_list=metrics,
    )
    logger.info(f"Scores: {scores}")


    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_file = output_dir_path / scores_filename

    with open(output_file, "w") as f:
        json.dump(scores, f, indent=4)
    
    logger.info(f"Scores saved to {output_file}")



if __name__ == "__main__":
    setup_logging('compare_output.log', 'INFO') # Change to 'DEBUG' for more verbose logging
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