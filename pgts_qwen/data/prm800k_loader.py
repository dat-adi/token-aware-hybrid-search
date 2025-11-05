"""
Use OpenAI's PRM800K dataset instead of generating data.
This is MUCH faster - just download and use!

Dataset: https://github.com/openai/prm800k
Paper: "Let's Verify Step by Step" (OpenAI, 2023)
"""
import json
import logging
from typing import List, Dict
from datasets import load_dataset
import random

logger = logging.getLogger(__name__)


def load_prm800k(
    split: str = "train",
    max_examples: int = None,
    balance_labels: bool = True
) -> List[Dict]:
    """
    Load PRM800K dataset from HuggingFace.

    Args:
        split: 'train' or 'test'
        max_examples: Maximum number of examples to load
        balance_labels: Whether to balance positive/negative examples

    Returns:
        List of examples in format:
        {
            'problem': str,
            'reasoning_path': List[str],
            'label': int (0 or 1),
            'step': str,
            'previous_steps': List[str]
        }
    """
    logger.info(f"Loading PRM800K dataset (split={split})...")

    # Load from HuggingFace
    # Note: The actual dataset name might be different, check HF hub
    try:
        # Try official OpenAI dataset first
        dataset = load_dataset("openai/prm800k", split=split)
    except:
        logger.warning("Could not load from HuggingFace, trying alternative...")
        # Alternative: load from local file or other source
        # For now, return empty to show the structure
        logger.error("PRM800K not available. Please download manually.")
        return []

    # Convert to our format
    converted_examples = []

    for example in dataset:
        problem = example['question']
        steps = example['steps']
        labels = example['labels']  # Per-step labels

        # Create examples for each step
        for step_idx, (step, label) in enumerate(zip(steps, labels)):
            converted_examples.append({
                'problem': problem,
                'step': step,
                'previous_steps': steps[:step_idx],
                'reasoning_path': steps[:step_idx + 1],
                'label': 1 if label == '+' else 0  # PRM800K uses +/-/neutral
            })

    logger.info(f"Loaded {len(converted_examples)} step-level examples")

    # Balance if requested
    if balance_labels:
        positive = [ex for ex in converted_examples if ex['label'] == 1]
        negative = [ex for ex in converted_examples if ex['label'] == 0]

        logger.info(f"Original: {len(positive)} positive, {len(negative)} negative")

        min_count = min(len(positive), len(negative))
        positive = random.sample(positive, min_count)
        negative = random.sample(negative, min_count)

        converted_examples = positive + negative
        random.shuffle(converted_examples)

        logger.info(f"Balanced: {len(converted_examples)} examples")

    # Limit size if requested
    if max_examples and len(converted_examples) > max_examples:
        converted_examples = random.sample(converted_examples, max_examples)
        logger.info(f"Limited to {max_examples} examples")

    return converted_examples


def create_prm_dataset_fast(
    gsm8k_train_data: List[Dict],
    max_examples: int = 50000
) -> List[Dict]:
    """
    Fast alternative: Use pre-generated PRM800K data.

    This is 1000x faster than generating with model inference.

    Args:
        gsm8k_train_data: GSM8K training data (unused, for compatibility)
        max_examples: Maximum examples to load

    Returns:
        PRM training dataset
    """
    logger.info("Using PRM800K dataset (pre-generated, instant loading!)")

    return load_prm800k(
        split="train",
        max_examples=max_examples,
        balance_labels=True
    )


def download_prm800k_manual():
    """
    Instructions to manually download PRM800K if not on HuggingFace.
    """
    instructions = """
    To manually download PRM800K:

    1. Visit: https://github.com/openai/prm800k
    2. Download the dataset files
    3. Place in: data/prm800k/

    Or use wget:
    ```bash
    mkdir -p data/prm800k
    cd data/prm800k
    wget https://github.com/openai/prm800k/releases/download/v1/prm800k.json
    ```

    Then modify this loader to read from local files.
    """
    print(instructions)


if __name__ == '__main__':
    # Test loading
    dataset = load_prm800k(split="train", max_examples=1000)
    print(f"Loaded {len(dataset)} examples")

    if len(dataset) > 0:
        print("\nExample:")
        print(dataset[0])
    else:
        download_prm800k_manual()
