"""
Use OpenAI's PRM800K dataset (Birchlabs stepwise-critic format).
This is MUCH faster - just download and use!

Dataset: https://huggingface.co/datasets/Birchlabs/openai-prm800k-stepwise-critic
Paper: "Let's Verify Step by Step" (OpenAI, 2023)

This dataset is pre-formatted for training step-wise critics (Process Reward Models).
Each example contains:
- A problem (instruction)
- Previous reasoning steps (responses)
- A candidate next step (next_response)
- Preference label indicating if this step was chosen (is_preferred_response)
"""
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
    Load PRM800K dataset from HuggingFace (Birchlabs stepwise-critic format).

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
    logger.info(f"Loading PRM800K dataset from Birchlabs/openai-prm800k-stepwise-critic (split={split})...")

    # Load from HuggingFace - Birchlabs reformatted dataset
    try:
        dataset = load_dataset("Birchlabs/openai-prm800k-stepwise-critic", split=split)
        logger.info(f"Successfully loaded {len(dataset)} examples from PRM800K")
    except Exception as e:
        logger.error(f"Failed to load PRM800K from HuggingFace: {e}")
        logger.error("Please ensure you have internet connection and access to:")
        logger.error("https://huggingface.co/datasets/Birchlabs/openai-prm800k-stepwise-critic")
        raise RuntimeError(f"Cannot load PRM800K dataset: {e}")

    # Convert to our format
    converted_examples = []

    logger.info(f"Processing {len(dataset)} examples from PRM800K...")

    for idx, example in enumerate(dataset):
        try:
            # Birchlabs stepwise-critic format:
            # - 'instruction': the problem/question
            # - 'responses': list of previous reasoning steps (already taken)
            # - 'next_response': the current step being evaluated
            # - 'is_preferred_response': whether this step was chosen (THIS IS OUR LABEL!)
            # - 'is_solution': whether this is the final solution step
            # - 'answer': final answer when is_solution=True (metadata only, don't use)
            # - 'rating': quality score (-1,0,1) - often nullable or all 1s, less useful
            # - 'is_human_response': ignore this field

            # Extract fields
            problem = example.get('instruction')
            previous_steps = example.get('responses', [])
            current_step = example.get('next_response', '')
            is_preferred = example.get('is_preferred_response')

            # Skip if missing essential fields
            if not problem or not current_step:
                if idx < 3:
                    logger.warning(f"Example {idx} missing instruction or next_response")
                continue

            # Skip if no preference label (this is our ground truth!)
            if is_preferred is None:
                continue

            # Convert to binary label
            # is_preferred_response tells us if this step was chosen over alternatives
            binary_label = 1 if is_preferred else 0

            # Ensure previous_steps is a list
            if previous_steps is None:
                previous_steps = []
            elif isinstance(previous_steps, str):
                previous_steps = [s.strip() for s in previous_steps.split('\n') if s.strip()]

            # Build the full reasoning path up to and including this step
            # NOTE: We always use next_response (even when is_solution=True)
            # The 'answer' field is just the extracted final answer, not a replacement
            reasoning_path = previous_steps + [current_step]

            # Create training example
            converted_examples.append({
                'problem': problem,
                'step': current_step,
                'previous_steps': previous_steps,
                'reasoning_path': reasoning_path,
                'label': binary_label
            })

        except Exception as e:
            if idx < 5:  # Log first few errors for debugging
                logger.warning(f"Error processing example {idx}: {e}")
            continue

    logger.info(f"Loaded {len(converted_examples)} step-level examples")

    if len(converted_examples) == 0:
        logger.error("No examples were successfully converted!")
        logger.error("Please check the dataset structure at:")
        logger.error("https://huggingface.co/datasets/Birchlabs/openai-prm800k-stepwise-critic")
        raise RuntimeError("Failed to convert any examples from PRM800K dataset")

    # Balance if requested
    if balance_labels:
        positive = [ex for ex in converted_examples if ex['label'] == 1]
        negative = [ex for ex in converted_examples if ex['label'] == 0]

        logger.info(f"Original: {len(positive)} positive, {len(negative)} negative")

        if len(positive) == 0 or len(negative) == 0:
            logger.warning(f"Cannot balance - missing positive or negative examples!")
            logger.warning(f"Positive: {len(positive)}, Negative: {len(negative)}")
        else:
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
    Load PRM800K dataset for training.

    This is 1000x faster than generating with model inference.

    Args:
        gsm8k_train_data: GSM8K training data (unused, for compatibility)
        max_examples: Maximum examples to load

    Returns:
        PRM training dataset from PRM800K
    """
    logger.info("Loading PRM800K dataset (pre-generated, instant loading!)")

    dataset = load_prm800k(
        split="train",
        max_examples=max_examples,
        balance_labels=True
    )

    return dataset


if __name__ == '__main__':
    # Test loading
    try:
        dataset = load_prm800k(split="train", max_examples=1000)
        print(f"Loaded {len(dataset)} examples")

        if len(dataset) > 0:
            print("\nExample structure:")
            
            print(dataset[0])

            # Show label distribution
            positive = sum(1 for ex in dataset if ex['label'] == 1)
            negative = sum(1 for ex in dataset if ex['label'] == 0)
            print(f"\nLabel distribution: {positive} positive, {negative} negative")
    except Exception as e:
        print(f"Error loading PRM800K: {e}")
        print("\nPlease ensure:")
        print("1. You have internet connection")
        print("2. You can access: https://huggingface.co/datasets/tasksource/PRM800K")
        print("3. The datasets library is installed: pip install datasets")
