"""
CACHED version of reward data generation - saves 20 hours on re-runs!
"""
import os
import json
import torch
from typing import List, Dict
import random
import logging
from tqdm import tqdm
import hashlib

from models.qwen3_wrapper import Qwen3ReasoningGenerator

logger = logging.getLogger(__name__)


class CachedRewardDatasetGenerator:
    """
    Generates PRM training data and caches it to disk.

    On first run: Takes 20 hours
    On subsequent runs: Takes 30 seconds!
    """

    def __init__(
        self,
        reasoning_generator: Qwen3ReasoningGenerator,
        cache_dir: str = "data/cached_prm",
        device: str = "cuda"
    ):
        self.reasoning_generator = reasoning_generator
        self.cache_dir = cache_dir
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)

    def create_prm_dataset(
        self,
        gsm8k_train_data: List[Dict],
        num_incorrect_per_problem: int = 3,
        max_examples: int = 50000,
        force_regenerate: bool = False
    ) -> List[Dict]:
        """
        Generate or load cached PRM dataset.

        Args:
            gsm8k_train_data: GSM8K training data
            num_incorrect_per_problem: Number of incorrect paths per problem
            max_examples: Maximum examples
            force_regenerate: If True, ignore cache and regenerate

        Returns:
            PRM training dataset
        """
        # Create cache key based on parameters
        cache_key = self._create_cache_key(
            len(gsm8k_train_data),
            num_incorrect_per_problem,
            max_examples
        )
        cache_file = os.path.join(self.cache_dir, f"prm_dataset_{cache_key}.json")

        # Try to load from cache
        if not force_regenerate and os.path.exists(cache_file):
            logger.info(f"Loading cached PRM dataset from {cache_file}...")
            try:
                with open(cache_file, 'r') as f:
                    dataset = json.load(f)
                logger.info(f"✓ Loaded {len(dataset)} examples from cache (instant!)")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Regenerating...")

        # Generate new dataset
        logger.info("No cache found. Generating PRM dataset (this will take ~20 hours)...")
        logger.info("💡 Tip: This will be cached for future runs!")

        dataset = self._generate_dataset(
            gsm8k_train_data,
            num_incorrect_per_problem,
            max_examples
        )

        # Save to cache
        logger.info(f"Saving dataset to cache: {cache_file}")
        try:
            with open(cache_file, 'w') as f:
                json.dump(dataset, f)
            logger.info(f"✓ Cached dataset saved! Future runs will be instant.")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        return dataset

    def _create_cache_key(
        self,
        num_problems: int,
        num_incorrect: int,
        max_examples: int
    ) -> str:
        """Create unique cache key from parameters."""
        key_str = f"{num_problems}_{num_incorrect}_{max_examples}"
        return hashlib.md5(key_str.encode()).hexdigest()[:8]

    def _generate_dataset(
        self,
        gsm8k_train_data: List[Dict],
        num_incorrect_per_problem: int,
        max_examples: int
    ) -> List[Dict]:
        """Generate dataset (slow operation)."""
        dataset = []

        for example in tqdm(gsm8k_train_data, desc="Generating PRM data"):
            problem = example['problem']
            correct_steps = example['steps']
            answer = example['answer']

            # Add correct steps
            for step_idx, step in enumerate(correct_steps):
                dataset.append({
                    'problem': problem,
                    'step': step,
                    'previous_steps': correct_steps[:step_idx],
                    'label': 1,
                    'reasoning_path': correct_steps[:step_idx + 1]
                })

            # Generate incorrect paths
            for _ in range(num_incorrect_per_problem):
                incorrect_path = self._generate_incorrect_path(problem, answer)

                if incorrect_path:
                    for step_idx, step in enumerate(incorrect_path):
                        dataset.append({
                            'problem': problem,
                            'step': step,
                            'previous_steps': incorrect_path[:step_idx],
                            'label': 0,
                            'reasoning_path': incorrect_path[:step_idx + 1]
                        })

            if len(dataset) >= max_examples:
                break

        # Balance dataset
        correct = [ex for ex in dataset if ex['label'] == 1]
        incorrect = [ex for ex in dataset if ex['label'] == 0]

        logger.info(f"Generated {len(correct)} correct, {len(incorrect)} incorrect")

        min_count = min(len(correct), len(incorrect))
        correct = random.sample(correct, min_count)
        incorrect = random.sample(incorrect, min_count)

        dataset = correct + incorrect
        random.shuffle(dataset)

        logger.info(f"Final dataset: {len(dataset)} examples")
        return dataset

    def _generate_incorrect_path(
        self,
        problem: str,
        correct_answer: str,
        max_steps: int = 8,
        max_attempts: int = 3
    ) -> List[str]:
        """Generate incorrect reasoning path (slow)."""
        for _ in range(max_attempts):
            reasoning_path = []
            original_temp = self.reasoning_generator.temperature
            self.reasoning_generator.temperature = 1.0

            try:
                for _ in range(max_steps):
                    generated_text, _ = self.reasoning_generator.generate_step(
                        problem,
                        reasoning_path,
                        return_hidden_states=False
                    )

                    if not generated_text:
                        break

                    reasoning_path.append(generated_text)

                    if "####" in generated_text or "answer is" in generated_text.lower():
                        break

                predicted_answer = self._extract_answer(reasoning_path)

                if predicted_answer and predicted_answer != correct_answer:
                    return reasoning_path

            except Exception as e:
                logger.warning(f"Error: {e}")
            finally:
                self.reasoning_generator.temperature = original_temp

        return []

    def _extract_answer(self, path: List[str]) -> str:
        """Extract answer from reasoning path."""
        if not path:
            return ""

        last = path[-1]
        if "####" in last:
            return last.split("####")[-1].strip()
        elif "answer is" in last.lower():
            parts = last.lower().split("answer is")
            if len(parts) > 1:
                return parts[-1].strip().split()[0]
        return ""


def create_prm_dataset_cached(
    gsm8k_train_data: List[Dict],
    reasoning_generator: Qwen3ReasoningGenerator,
    num_incorrect_per_problem: int = 3,
    max_examples: int = 50000,
    cache_dir: str = "data/cached_prm",
    force_regenerate: bool = False
) -> List[Dict]:
    """
    Create PRM dataset with caching.

    First run: ~20 hours
    Subsequent runs: ~30 seconds!
    """
    generator = CachedRewardDatasetGenerator(
        reasoning_generator,
        cache_dir=cache_dir
    )

    return generator.create_prm_dataset(
        gsm8k_train_data,
        num_incorrect_per_problem,
        max_examples,
        force_regenerate
    )
