"""
GSM8k dataset loader and preprocessor.
"""
from datasets import load_dataset
from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class GSM8kDataset:
    """
    GSM8k dataset loader and processor.

    Processing:
        1. Load train split (7,473 examples)
        2. Extract problem, solution steps, final answer
        3. Parse answer from '#### {number}' format
        4. Create train/val split (90/10)
    """

    def __init__(self, train_split_ratio: float = 0.9):
        """
        Initialize GSM8k dataset.

        Args:
            train_split_ratio: Ratio of data to use for training
        """
        self.train_split_ratio = train_split_ratio
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def load_gsm8k(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load and preprocess GSM8k dataset.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading GSM8k dataset...")

        # Load from HuggingFace datasets
        dataset = load_dataset("gsm8k", "main")

        # Process train split
        train_full = self._process_split(dataset['train'])

        # Split into train/val
        split_idx = int(len(train_full) * self.train_split_ratio)
        self.train_data = train_full[:split_idx]
        self.val_data = train_full[split_idx:]

        # Process test split
        self.test_data = self._process_split(dataset['test'])

        logger.info(f"Loaded {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test examples")

        return self.train_data, self.val_data, self.test_data

    def _process_split(self, split) -> List[Dict]:
        """
        Process a dataset split.

        Args:
            split: HuggingFace dataset split

        Returns:
            List of processed examples
        """
        processed = []

        for example in split:
            problem = example['question']
            solution = example['answer']

            # Extract final answer (format: "#### {number}")
            answer = self._extract_answer(solution)

            # Parse solution into steps
            steps = self._parse_solution_steps(solution)

            processed.append({
                'problem': problem,
                'solution': solution,
                'steps': steps,
                'answer': answer
            })

        return processed

    def _extract_answer(self, solution: str) -> str:
        """
        Extract final answer from solution.

        Args:
            solution: Full solution text

        Returns:
            Final answer string
        """
        if "####" in solution:
            answer = solution.split("####")[-1].strip()
            return answer
        else:
            logger.warning("No #### found in solution, extracting last number")
            # Try to find last number
            numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', solution)
            if numbers:
                return numbers[-1]
            return ""

    def _parse_solution_steps(self, solution: str) -> List[str]:
        """
        Parse solution into individual reasoning steps.

        Args:
            solution: Full solution text

        Returns:
            List of reasoning steps
        """
        # Split at answer marker
        if "####" in solution:
            reasoning = solution.split("####")[0].strip()
        else:
            reasoning = solution.strip()

        # Split into sentences/steps
        # GSM8k solutions are typically sentence-based
        steps = []

        # Split by periods, but keep calculations together
        current_step = ""
        for char in reasoning:
            current_step += char
            if char == '.' and len(current_step.strip()) > 10:
                # End of step
                step = current_step.strip()
                if step:
                    steps.append(step)
                current_step = ""

        # Add remaining
        if current_step.strip():
            steps.append(current_step.strip())

        return steps

    def get_train_data(self) -> List[Dict]:
        """Get training data."""
        return self.train_data

    def get_val_data(self) -> List[Dict]:
        """Get validation data."""
        return self.val_data

    def get_test_data(self) -> List[Dict]:
        """Get test data."""
        return self.test_data

    def get_example(self, split: str, idx: int) -> Dict:
        """
        Get specific example.

        Args:
            split: 'train', 'val', or 'test'
            idx: Example index

        Returns:
            Example dictionary
        """
        if split == 'train':
            return self.train_data[idx]
        elif split == 'val':
            return self.val_data[idx]
        elif split == 'test':
            return self.test_data[idx]
        else:
            raise ValueError(f"Unknown split: {split}")


def load_gsm8k() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Convenience function to load GSM8k dataset.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    dataset = GSM8kDataset()
    return dataset.load_gsm8k()
