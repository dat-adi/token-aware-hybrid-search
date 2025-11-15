"""
Format GSM8K-Gen-Stepwise dataset for PRM training.

Dataset: https://huggingface.co/datasets/ebony59/gsm8k-gen-stepwise
Converts multiple-completion format to step-wise reward model training format.
"""
import re
import random
import logging
from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GSM8KStepwiseFormatter:
    """
    Format GSM8K-Gen-Stepwise dataset for Process Reward Model training.

    Input format (from Hugging Face dataset):
        - prompt: Problem text with special tokens
        - completions: List of solution approaches (variable length)
        - labels: List of boolean labels (True=correct, False=incorrect)

    Output format (for PRM training):
        - problem: Clean problem text
        - reasoning_path: List of reasoning steps
        - label: 0 (incorrect) or 1 (correct)
        - step: Individual step text
        - previous_steps: Steps before this one
    """

    def __init__(self, seed: int = 42):
        """
        Initialize formatter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def load_and_format(
        self,
        dataset_name: str = "ebony59/gsm8k-gen-stepwise",
        max_examples: int = None,
        balance_dataset: bool = True
    ) -> List[Dict]:
        """
        Load and format the dataset for PRM training.

        Args:
            dataset_name: HuggingFace dataset identifier
            max_examples: Maximum number of examples to process (None = all)
            balance_dataset: Whether to balance correct/incorrect examples

        Returns:
            List of formatted PRM training examples
        """
        logger.info(f"Loading dataset: {dataset_name}")

        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, split="train")

        if max_examples:
            logger.info(f"Limiting to {max_examples} examples")
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        logger.info(f"Processing {len(dataset)} examples...")

        formatted_examples = []

        for example in tqdm(dataset, desc="Formatting dataset"):
            # Extract fields
            prompt = example['prompt']
            completions = example['completions']
            labels = example['labels']

            # Clean problem text (remove special tokens)
            problem = self._clean_problem_text(prompt)

            # Process each completion
            for completion, is_correct in zip(completions, labels):
                # Parse completion into steps
                steps = self._parse_completion_steps(completion)

                if not steps:
                    continue

                # Convert boolean to int label
                label = 1 if is_correct else 0

                # Create step-wise examples
                step_examples = self._create_stepwise_examples(
                    problem=problem,
                    steps=steps,
                    label=label
                )

                formatted_examples.extend(step_examples)

        logger.info(f"Generated {len(formatted_examples)} step-wise examples")

        # Balance dataset if requested
        if balance_dataset:
            formatted_examples = self._balance_dataset(formatted_examples)

        # Shuffle
        random.shuffle(formatted_examples)

        logger.info(f"✓ Final dataset: {len(formatted_examples)} examples")

        return formatted_examples

    def _clean_problem_text(self, prompt: str) -> str:
        """
        Clean problem text by removing special tokens.

        Args:
            prompt: Raw prompt with special tokens like <|user|>, <|end|>

        Returns:
            Clean problem text
        """
        # Remove special tokens
        text = prompt.replace('<|user|>', '')
        text = text.replace('<|end|>', '')
        text = text.replace('<|assistant|>', '')
        text = text.strip()

        return text

    def _parse_completion_steps(self, completion: str) -> List[str]:
        """
        Parse completion into individual reasoning steps.

        GSM8K format uses:
        - Sentences ending with periods
        - Calculations in <<...>> notation
        - Final answer marked with ####

        Args:
            completion: Full solution text

        Returns:
            List of reasoning steps
        """
        # Remove answer marker for step parsing
        if '####' in completion:
            reasoning_part = completion.split('####')[0].strip()
            answer_part = completion.split('####')[-1].strip()
        else:
            reasoning_part = completion.strip()
            answer_part = None

        steps = []

        # Split by sentences, keeping calculations together
        # Pattern: sentence ending with period, or calculation <<...>>
        current_step = ""

        for line in reasoning_part.split('\n'):
            line = line.strip()
            if not line:
                continue

            # If line contains calculation notation <<...>>
            if '<<' in line and '>>' in line:
                # Keep calculation with its sentence
                current_step += ' ' + line

                # If sentence ends after calculation, finalize step
                if line.rstrip().endswith('.'):
                    steps.append(current_step.strip())
                    current_step = ""
            else:
                # Regular sentence
                current_step += ' ' + line

                # Check if sentence ends
                if line.rstrip().endswith('.'):
                    steps.append(current_step.strip())
                    current_step = ""

        # Add remaining content
        if current_step.strip():
            steps.append(current_step.strip())

        # Add final answer as last step if present
        if answer_part:
            steps.append(f"#### {answer_part}")

        # Filter out empty steps
        steps = [s for s in steps if s.strip()]

        return steps

    def _create_stepwise_examples(
        self,
        problem: str,
        steps: List[str],
        label: int
    ) -> List[Dict]:
        """
        Create step-wise training examples.

        Each step in the reasoning path becomes a separate example,
        with the label propagated to all steps.

        Args:
            problem: Problem text
            steps: List of reasoning steps
            label: 0 (incorrect) or 1 (correct)

        Returns:
            List of step-wise examples
        """
        examples = []

        for step_idx, step in enumerate(steps):
            example = {
                'problem': problem,
                'step': step,
                'previous_steps': steps[:step_idx],
                'reasoning_path': steps[:step_idx + 1],
                'label': label,
                'step_index': step_idx,
                'total_steps': len(steps)
            }
            examples.append(example)

        return examples

    def _balance_dataset(self, examples: List[Dict]) -> List[Dict]:
        """
        Balance dataset to have equal correct/incorrect examples.

        Args:
            examples: List of training examples

        Returns:
            Balanced dataset
        """
        correct = [ex for ex in examples if ex['label'] == 1]
        incorrect = [ex for ex in examples if ex['label'] == 0]

        logger.info(f"Before balancing: {len(correct)} correct, {len(incorrect)} incorrect")

        # Balance to smaller count
        min_count = min(len(correct), len(incorrect))

        if len(correct) > min_count:
            correct = random.sample(correct, min_count)

        if len(incorrect) > min_count:
            incorrect = random.sample(incorrect, min_count)

        balanced = correct + incorrect

        logger.info(f"After balancing: {len(correct)} correct, {len(incorrect)} incorrect")

        return balanced

    def save_to_jsonl(self, examples: List[Dict], output_path: str):
        """
        Save formatted examples to JSONL file.

        Args:
            examples: Formatted examples
            output_path: Output file path
        """
        import json

        logger.info(f"Saving {len(examples)} examples to {output_path}")

        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

        logger.info(f"✓ Saved to {output_path}")

    def create_train_val_split(
        self,
        examples: List[Dict],
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split examples into train and validation sets.

        Args:
            examples: All examples
            val_ratio: Ratio of data for validation

        Returns:
            Tuple of (train_examples, val_examples)
        """
        random.shuffle(examples)

        split_idx = int(len(examples) * (1 - val_ratio))

        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} val")

        return train_examples, val_examples


def format_gsm8k_stepwise(
    dataset_name: str = "ebony59/gsm8k-gen-stepwise",
    max_examples: int = None,
    balance_dataset: bool = True,
    val_ratio: float = 0.1,
    output_dir: str = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience function to format GSM8K-Gen-Stepwise dataset.

    Args:
        dataset_name: HuggingFace dataset identifier
        max_examples: Maximum examples to process
        balance_dataset: Whether to balance correct/incorrect
        val_ratio: Validation split ratio
        output_dir: Optional directory to save JSONL files

    Returns:
        Tuple of (train_examples, val_examples)
    """
    formatter = GSM8KStepwiseFormatter()

    # Load and format
    examples = formatter.load_and_format(
        dataset_name=dataset_name,
        max_examples=max_examples,
        balance_dataset=balance_dataset
    )

    # Split train/val
    train_examples, val_examples = formatter.create_train_val_split(
        examples,
        val_ratio=val_ratio
    )

    # Save to disk if output_dir specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

        formatter.save_to_jsonl(
            train_examples,
            os.path.join(output_dir, 'prm_train.jsonl')
        )
        formatter.save_to_jsonl(
            val_examples,
            os.path.join(output_dir, 'prm_val.jsonl')
        )

    return train_examples, val_examples


if __name__ == "__main__":
    """
    Example usage:

    python data/gsm8k_stepwise_formatter.py
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Format GSM8K-Gen-Stepwise dataset for PRM training"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ebony59/gsm8k-gen-stepwise',
        help='HuggingFace dataset name'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=None,
        help='Maximum examples to process (default: all)'
    )
    parser.add_argument(
        '--no_balance',
        action='store_true',
        help='Do not balance correct/incorrect examples'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/prm_formatted',
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Format dataset
    train_examples, val_examples = format_gsm8k_stepwise(
        dataset_name=args.dataset,
        max_examples=args.max_examples,
        balance_dataset=not args.no_balance,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir
    )

    logger.info("=" * 60)
    logger.info("Dataset formatting complete!")
    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Val examples: {len(val_examples)}")

    # Show example
    if train_examples:
        logger.info("\nExample training instance:")
        example = train_examples[0]
        logger.info(f"Problem: {example['problem'][:100]}...")
        logger.info(f"Step: {example['step'][:100]}...")
        logger.info(f"Label: {example['label']}")
        logger.info(f"Previous steps: {len(example['previous_steps'])}")
