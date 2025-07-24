from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration class for loading and processing a dataset."""
    hf_path: str  # Path to the dataset on Hugging Face Hub (e.g., "truthful_qa")
    hf_name: str | None  # Specific configuration name of the dataset (e.g., "generation")
    split: str  # The dataset split to use (e.g., "train", "validation")
    text_column: str  # The name of the column containing the main text/prompt
    template_with_task_hints: str  # A format string for when task hints are enabled
    max_text_length_filter: int | None  # Optional max character length to filter samples
    task_system_prompt: str  # A generic system prompt describing the task (e.g., summarization)

    



DATASET_CONFIGS = {
    "truthfulqa": DatasetConfig(
        hf_path="truthful_qa",
        hf_name="generation",
        split="validation",
        text_column="question",
        template_with_task_hints="Question: {text} Answer: ",
        max_text_length_filter=None,
        task_system_prompt="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer."
    ),
    "triviaqa": DatasetConfig(
        hf_path="mandarjoshi/trivia_qa",
        hf_name="rc.nocontext",
        split="train",
        text_column="question",
        template_with_task_hints="Question: {text} Answer: ",
        max_text_length_filter=None,
        task_system_prompt="You are a question-answering AI assistant. You will receive the question and you have to reply directly with the answer."
    ),
    "cnn_dailymail": DatasetConfig(
        hf_path="cnn_dailymail",
        hf_name="3.0.0",
        split="train",
        text_column="article",
        template_with_task_hints="CNN Article: {text} Summary of the article: ",
        max_text_length_filter=800,
        task_system_prompt="You are a summarization AI assistant. You will receive a CNN daily mail article and you will reply directly with the summary."

    ),
    "samsum": DatasetConfig(
        hf_path="knkarthick/samsum",
        hf_name=None,
        split="train",
        text_column="dialogue",
        template_with_task_hints="Dialogue: {text} Summary of the dialogue: ",
        max_text_length_filter=800,
        task_system_prompt="You are a summarization AI assistant. You will receive a messenger-like conversation and you will reply directly with the summary."
    ),
}

def get_dataset_config(name: str) -> DatasetConfig:
    """Retrieves the configuration for a given dataset name."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset configuration: {name}")
    return DATASET_CONFIGS[name]