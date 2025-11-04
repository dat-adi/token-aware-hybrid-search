"""
Process Reward Model for step-level reasoning verification.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ProcessRewardModel(nn.Module):
    """
    Step-level reward model for reasoning verification.

    Training:
        - Dataset: GSM8k with step annotations
        - Labels: Binary (correct=1, incorrect=0) per step
        - Loss: Binary cross-entropy
        - Base: Qwen3-1.7B or Qwen3-4B

    Input:
        - problem: Original math problem text
        - reasoning_path: List of reasoning steps

    Output:
        - step_rewards: [num_steps] scores in [0, 1]
        - final_correctness: Binary prediction for solution
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        num_labels: int = 2,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        dropout: float = 0.1
    ):
        """
        Initialize Process Reward Model.

        Args:
            model_name: Base model identifier
            num_labels: Number of classification labels (2 for binary)
            device: Device to load model on
            torch_dtype: Data type for model weights
            dropout: Dropout probability
        """
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.num_labels = num_labels

        logger.info(f"Loading reward model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch_dtype,
            device_map="auto",
            problem_type="single_label_classification"
        )

        logger.info("Reward model loaded successfully")

    def format_input(
        self,
        problem: str,
        reasoning_path: List[str]
    ) -> str:
        """
        Format problem and reasoning path for model input.

        Args:
            problem: Original problem text
            reasoning_path: List of reasoning steps

        Returns:
            Formatted input string
        """
        steps_text = "\n".join(
            f"Step {i+1}: {step}"
            for i, step in enumerate(reasoning_path)
        )

        formatted = f"""Problem: {problem}

Solution:
{steps_text}"""

        return formatted

    def compute_step_reward(
        self,
        problem: str,
        reasoning_path: List[str],
        step_idx: int
    ) -> float:
        """
        Compute reward for a single step.

        Args:
            problem: Original problem
            reasoning_path: Full reasoning path up to step
            step_idx: Index of step to evaluate

        Returns:
            Reward score in [0, 1]
        """
        # Evaluate the partial path up to and including this step
        partial_path = reasoning_path[:step_idx + 1]
        input_text = self.format_input(problem, partial_path)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Probability of correct class (label=1)
            reward = probs[0, 1].item()

        return reward

    def compute_step_rewards(
        self,
        problem: str,
        reasoning_path: List[str]
    ) -> List[float]:
        """
        Compute rewards for all steps in reasoning path.

        Args:
            problem: Original problem
            reasoning_path: Complete reasoning path

        Returns:
            List of reward scores, one per step
        """
        rewards = []
        for i in range(len(reasoning_path)):
            reward = self.compute_step_reward(problem, reasoning_path, i)
            rewards.append(reward)

        return rewards

    def evaluate_solution(
        self,
        problem: str,
        reasoning_path: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate complete solution.

        Args:
            problem: Original problem
            reasoning_path: Complete reasoning path

        Returns:
            Dictionary with step rewards and final correctness
        """
        if len(reasoning_path) == 0:
            return {
                'step_rewards': [],
                'final_correctness': 0.0,
                'mean_reward': 0.0
            }

        step_rewards = self.compute_step_rewards(problem, reasoning_path)

        return {
            'step_rewards': step_rewards,
            'final_correctness': step_rewards[-1],  # Last step reward
            'mean_reward': sum(step_rewards) / len(step_rewards)
        }

    def batch_evaluate(
        self,
        problems: List[str],
        reasoning_paths: List[List[str]]
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple solutions in batch.

        Args:
            problems: List of problems
            reasoning_paths: List of reasoning paths

        Returns:
            List of evaluation dictionaries
        """
        results = []
        for problem, path in zip(problems, reasoning_paths):
            result = self.evaluate_solution(problem, path)
            results.append(result)

        return results

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for training.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels

        Returns:
            Model outputs with loss
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def save_pretrained(self, save_path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> 'ProcessRewardModel':
        """
        Load pretrained reward model.

        Args:
            model_path: Path to saved model
            device: Device to load on
            torch_dtype: Data type

        Returns:
            Loaded ProcessRewardModel
        """
        instance = cls.__new__(cls)
        super(ProcessRewardModel, instance).__init__()

        instance.device = device
        instance.model_path = model_path

        logger.info(f"Loading reward model from {model_path}")

        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token

        instance.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto"
        )

        instance.model.eval()
        logger.info("Reward model loaded successfully")

        return instance


def create_reward_training_example(
    problem: str,
    reasoning_steps: List[str],
    label: int
) -> Dict[str, any]:
    """
    Create training example for reward model.

    Args:
        problem: Math problem
        reasoning_steps: Reasoning steps
        label: 0 (incorrect) or 1 (correct)

    Returns:
        Training example dictionary
    """
    steps_text = "\n".join(
        f"Step {i+1}: {step}"
        for i, step in enumerate(reasoning_steps)
    )

    text = f"""Problem: {problem}

Solution:
{steps_text}"""

    return {
        'text': text,
        'label': label,
        'problem': problem,
        'steps': reasoning_steps
    }
