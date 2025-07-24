import logging
from functools import partial

import datasets
from sklearn.model_selection import train_test_split

from .config import DatasetConfig, get_dataset_config

logger = logging.getLogger(__name__)

def _apply_task_prompt_template(
    examples: dict[str, list], 
    config: DatasetConfig, 
    use_task_hints: bool
) -> dict[str, list[str]]:
    """
    Applies a task-specific prompt template to a batch of examples.

    This function formats each text entry based on whether task hints are enabled.

    Args:
        examples (dict[str, list]): A batch of examples from a Hugging Face dataset.
        config (DatasetConfig): Configuration object for the dataset.
        use_task_hints (bool): If True, applies the template with hints; otherwise,
                               returns the text content directly.

    Returns:
        (dict[str, list[str]]): A dictionary with a new "processed_text" key
                                containing the formatted texts.
    """
    texts = examples[config.text_column]
    processed_texts = []
    for text_content in texts:
        text_content = str(text_content).strip()
        if use_task_hints:
            processed_texts.append(config.template_with_task_hints.format(text=text_content))
        else:
            processed_texts.append(text_content)
    return {"processed_text": processed_texts}


def _filter_by_length(example: dict, config: DatasetConfig) -> bool:
    """Filters a dataset example based on the configured maximum text length."""
    if config.max_text_length_filter is None:
        return True
    
    text_content = example[config.text_column]
    if isinstance(text_content, str):
        return len(text_content) <= config.max_text_length_filter
    logger.warning(f"Encountered non-string data in text_column '{config.text_column}' during length filtering: {type(text_content)}. Keeping example.")
    return True


def load_and_prepare_dataset(
    dataset_name: str,
    dataset_size: int,
    use_task_hints: bool,
    seed: int,
    split_ratio: float = 0.8,
) -> tuple[list[str], list[str], str]:
    """
    Loads, processes, shuffles, crops, and splits a dataset based on its configuration.

    Args:
        dataset_name (str): The short name of the dataset (e.g., "truthfulqa").
        dataset_size (int): The total number of samples to use from the dataset.
        use_task_hints (bool): Whether to apply task-specific hints to the user prompts.
        seed (int): The random seed for shuffling and splitting.
        split_ratio (float): The proportion of the dataset to allocate to the training set.

    Returns:
        (tuple[list[str], list[str], str]): A tuple containing:
            - A list of processed training texts.
            - A list of processed testing texts.
            - The task-specific system prompt string from the dataset config.
    """
    try:
        config = get_dataset_config(dataset_name)
        logger.debug(f"Loading dataset: {dataset_name} with path='{config.hf_path}', name='{config.hf_name}', split='{config.split}'")
    except ValueError as e:
        logger.error(f"Failed to get dataset config: {e}")
        raise


    # 1. Load dataset
    try:
        raw_dataset = datasets.load_dataset(
            path=config.hf_path,
            name=config.hf_name,
            split=config.split,
        )
        logger.debug(f"Successfully loaded raw dataset. Original size: {len(raw_dataset)}")
    except Exception as e:
        logger.exception(f"Failed to load dataset '{dataset_name}' from Hugging Face.")
        raise

    # 2. Filter by length (if set)
    if config.max_text_length_filter is not None:
        logger.debug(f"Filtering dataset by max text length: {config.max_text_length_filter} for column '{config.text_column}'")
        original_len = len(raw_dataset)
        filter_fn_with_config = lambda example: _filter_by_length(example, config)
        filtered_dataset = raw_dataset.filter(filter_fn_with_config)
        logger.debug(f"Filtered dataset from {original_len} to {len(filtered_dataset)} examples.")
        if len(filtered_dataset) == 0 :
            logger.error(f"Dataset became empty after length filtering for {dataset_name}. Check filter criteria and data.")
            raise ValueError(f"Dataset empty after filtering for {dataset_name}.")
        raw_dataset = filtered_dataset

    # 3. Shuffle
    logger.debug(f"Shuffling dataset with seed: {seed}")
    shuffled_dataset = raw_dataset.shuffle(seed=seed)

    # 4. Crop to size
    current_size = len(shuffled_dataset)
    if current_size < dataset_size:
        logger.warning(f"Requested dataset_size ({dataset_size}) is larger than available shuffled and filtered data ({current_size}). Using all available data.")
        effective_size = current_size
    else:
        effective_size = dataset_size
    
    if current_size > effective_size:
        logger.debug(f"Cropping dataset from {current_size} to {effective_size} examples.")
        cropped_dataset = shuffled_dataset.select(range(effective_size))
    else:
        cropped_dataset = shuffled_dataset
    
    if len(cropped_dataset) == 0:
        logger.error(f"Dataset empty after cropping for {dataset_name}. Effective size was {effective_size}.")
        raise ValueError(f"Dataset empty after cropping for {dataset_name}.")
    
    # 5. Preprocess (apply task prompt template if required)
    logger.debug(f"Applying task prompt template. Using task hints: {use_task_hints}")
    map_fn_with_args = partial(_apply_task_prompt_template, config=config, use_task_hints=use_task_hints)
    processed_dataset = cropped_dataset.map(
        map_fn_with_args,
        batched=True,
        remove_columns=cropped_dataset.column_names
    )


    all_processed_texts = processed_dataset["processed_text"]
    train_texts, test_texts = train_test_split(
        all_processed_texts,
        train_size=split_ratio,
        random_state=seed,
        shuffle=False
    )
    logger.info(f"Dataset split into training ({len(train_texts)} samples) and testing ({len(test_texts)} samples).")

    return train_texts, test_texts, config.task_system_prompt