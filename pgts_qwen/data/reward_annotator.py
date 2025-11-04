"""
Generate step-level annotations for Process Reward Model training.
"""
import torch
from typing import List, Dict, Tuple
import random
import logging
from tqdm import tqdm

from ..models.qwen3_wrapper import Qwen3ReasoningGenerator

logger = logging.getLogger(__name__)


class RewardDatasetGenerator:
    """
    Generate step-level annotations for PRM training.

    Strategy:
        1. For each GSM8k problem:
            a. Parse ground truth solution into steps
            b. Label all steps as correct (label=1)

        2. Generate incorrect trajectories:
            a. Sample from base Qwen3 with high temperature
            b. Select paths leading to wrong answers
            c. Label steps in wrong paths as incorrect (label=0)

        3. Balance dataset: 50% correct, 50% incorrect steps
    """

    def __init__(
        self,
        reasoning_generator: Qwen3ReasoningGenerator,
        device: str = "cuda"
    ):
        """
        Initialize dataset generator.

        Args:
            reasoning_generator: Qwen3 model for generating incorrect steps
            device: Device to use
        """
        self.reasoning_generator = reasoning_generator
        self.device = device

    def create_prm_dataset(
        self,
        gsm8k_train_data: List[Dict],
        num_incorrect_per_problem: int = 3,
        max_examples: int = 50000
    ) -> List[Dict]:
        """
        Generate step-level annotations for PRM training.

        Args:
            gsm8k_train_data: GSM8k training data
            num_incorrect_per_problem: Number of incorrect paths to generate per problem
            max_examples: Maximum number of examples

        Returns:
            List of training examples with step-level labels
        """
        logger.info(f"Generating PRM training data from {len(gsm8k_train_data)} problems...")

        dataset = []

        for example in tqdm(gsm8k_train_data, desc="Generating PRM data"):
            problem = example['problem']
            correct_steps = example['steps']
            answer = example['answer']

            # Add correct steps
            for step_idx, step in enumerate(correct_steps):
                partial_steps = correct_steps[:step_idx + 1]

                dataset.append({
                    'problem': problem,
                    'step': step,
                    'previous_steps': correct_steps[:step_idx],
                    'label': 1,  # Correct
                    'reasoning_path': partial_steps
                })

            # Generate incorrect trajectories
            for _ in range(num_incorrect_per_problem):
                incorrect_path = self._generate_incorrect_path(problem, answer)

                if incorrect_path:
                    # Label steps in incorrect path
                    for step_idx, step in enumerate(incorrect_path):
                        dataset.append({
                            'problem': problem,
                            'step': step,
                            'previous_steps': incorrect_path[:step_idx],
                            'label': 0,  # Incorrect
                            'reasoning_path': incorrect_path[:step_idx + 1]
                        })

            # Check if we've reached max examples
            if len(dataset) >= max_examples:
                break

        # Balance dataset
        correct_examples = [ex for ex in dataset if ex['label'] == 1]
        incorrect_examples = [ex for ex in dataset if ex['label'] == 0]

        logger.info(f"Generated {len(correct_examples)} correct and {len(incorrect_examples)} incorrect examples")

        # Balance to 50/50
        min_count = min(len(correct_examples), len(incorrect_examples))
        correct_examples = random.sample(correct_examples, min_count)
        incorrect_examples = random.sample(incorrect_examples, min_count)

        balanced_dataset = correct_examples + incorrect_examples
        random.shuffle(balanced_dataset)

        logger.info(f"Final balanced dataset: {len(balanced_dataset)} examples")

        return balanced_dataset

    def _generate_incorrect_path(
        self,
        problem: str,
        correct_answer: str,
        max_steps: int = 8,
        max_attempts: int = 3
    ) -> List[str]:
        """
        Generate an incorrect reasoning path.

        Args:
            problem: Math problem
            correct_answer: Correct answer
            max_steps: Maximum steps to generate
            max_attempts: Maximum attempts to generate wrong answer

        Returns:
            List of reasoning steps (empty if failed)
        """
        for _ in range(max_attempts):
            reasoning_path = []

            # Generate steps with high temperature (more diversity)
            original_temp = self.reasoning_generator.temperature
            self.reasoning_generator.temperature = 1.0

            try:
                for step_num in range(max_steps):
                    # Generate next step
                    generated_text, _ = self.reasoning_generator.generate_step(
                        problem,
                        reasoning_path,
                        return_hidden_states=False
                    )

                    if not generated_text:
                        break

                    reasoning_path.append(generated_text)

                    # Check if answer is present
                    if "####" in generated_text or "answer is" in generated_text.lower():
                        break

                # Extract answer from path
                predicted_answer = self._extract_answer_from_path(reasoning_path)

                # Check if it's wrong
                if predicted_answer and predicted_answer != correct_answer:
                    self.reasoning_generator.temperature = original_temp
                    return reasoning_path

            except Exception as e:
                logger.warning(f"Error generating incorrect path: {e}")

            finally:
                self.reasoning_generator.temperature = original_temp

        return []

    def _extract_answer_from_path(self, reasoning_path: List[str]) -> str:
        """
        Extract answer from reasoning path.

        Args:
            reasoning_path: List of reasoning steps

        Returns:
            Extracted answer or empty string
        """
        # Check last step
        if len(reasoning_path) == 0:
            return ""

        last_step = reasoning_path[-1]

        if "####" in last_step:
            answer = last_step.split("####")[-1].strip()
            return answer
        elif "answer is" in last_step.lower():
            parts = last_step.lower().split("answer is")
            if len(parts) > 1:
                answer = parts[-1].strip().split()[0]
                return answer

        return ""


def create_prm_dataset(
    gsm8k_train_data: List[Dict],
    reasoning_generator: Qwen3ReasoningGenerator,
    num_incorrect_per_problem: int = 3,
    max_examples: int = 50000
) -> List[Dict]:
    """
    Convenience function to generate PRM dataset.

    Args:
        gsm8k_train_data: GSM8k training data
        reasoning_generator: Qwen3 model
        num_incorrect_per_problem: Number of incorrect paths per problem
        max_examples: Maximum examples

    Returns:
        PRM training dataset
    """
    generator = RewardDatasetGenerator(reasoning_generator)
    return generator.create_prm_dataset(
        gsm8k_train_data,
        num_incorrect_per_problem,
        max_examples
    )
