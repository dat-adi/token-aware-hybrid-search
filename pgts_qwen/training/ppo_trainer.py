"""
PPO trainer for policy network.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from models.policy_network import GPSPolicyNetwork
from models.qwen3_wrapper import Qwen3ReasoningGenerator
from models.reward_model import ProcessRewardModel
from tree_search.search_algorithm import PGTSSearch, SearchConfig, SearchTrajectory

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO training configuration."""
    learning_rate: float = 1e-5
    batch_size: int = 32
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    discount_gamma: float = 0.99
    max_grad_norm: float = 1.0
    final_reward_correct: float = 10.0
    final_reward_incorrect: float = -5.0
    step_penalty: float = -0.1


class PPOTrainer:
    """
    PPO training for policy network.

    Hyperparameters:
        - Learning rate: 1e-5
        - Batch size: 32 problems
        - Rollout length: Complete searches
        - PPO epochs: 4
        - Clip epsilon: 0.2
        - Value coefficient: 0.5
        - Entropy coefficient: 0.01
        - GAE lambda: 0.95
        - Discount gamma: 0.99

    Training Procedure:
        1. Collect trajectories using current policy
        2. Compute advantages using GAE
        3. Update policy with PPO objective
        4. Update value network (for baseline)
        5. Log metrics (reward, accuracy, token usage)

    Reward Function:
        - Intermediate rewards: PRM scores
        - Final reward: +10 if correct, -5 if incorrect
        - Step penalty: -0.1 per node (efficiency incentive)
    """

    def __init__(
        self,
        policy_network: GPSPolicyNetwork,
        reasoning_generator: Qwen3ReasoningGenerator,
        reward_model: ProcessRewardModel,
        config: Optional[PPOConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_network: GPS policy network
            reasoning_generator: Qwen3 model
            reward_model: Process reward model
            config: PPO configuration
            device: Device to use
        """
        self.policy_network = policy_network.to(device)
        self.reasoning_generator = reasoning_generator
        self.reward_model = reward_model
        self.config = config or PPOConfig()
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=float(self.config.learning_rate)
        )

        # Search algorithm
        search_config = SearchConfig(device=device)
        self.search_algorithm = PGTSSearch(
            reasoning_generator=reasoning_generator,
            reward_model=reward_model,
            policy_network=policy_network,
            config=search_config
        )

        # Action statistics tracking
        self.action_history = {
            'iterations': [],
            'action_counts': [],  # List of dicts with action counts per iteration
            'action_distributions': []  # List of dicts with action percentages per iteration
        }

        logger.info("PPO Trainer initialized")

    def collect_trajectories(
        self,
        problems: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> List[SearchTrajectory]:
        """
        Collect trajectories using current policy.

        Args:
            problems: List of problems to solve
            ground_truths: Ground truth answers (for evaluation)

        Returns:
            List of trajectories
        """
        trajectories = []

        for idx, problem in enumerate(tqdm(problems, desc="Collecting trajectories")):
            # Run search
            tree, trajectory = self.search_algorithm.search(
                problem,
                collect_trajectory=True
            )

            # Evaluate if ground truth provided
            if ground_truths and idx < len(ground_truths):
                trajectory.is_correct = self.search_algorithm.evaluate_solution(
                    trajectory,
                    ground_truths[idx]
                )

            trajectories.append(trajectory)

        return trajectories

    def compute_action_statistics(
        self,
        trajectories: List[SearchTrajectory]
    ) -> Dict[str, any]:
        """
        Compute action statistics from trajectories.

        Args:
            trajectories: List of search trajectories

        Returns:
            Dictionary with action counts and distributions
        """
        action_names = {
            0: "EXPAND",
            1: "BRANCH",
            2: "BACKTRACK",
            3: "TERMINATE",
            4: "SPAWN"
        }

        # Count actions
        action_counts = {name: 0 for name in action_names.values()}

        for trajectory in trajectories:
            for action in trajectory.actions:
                action_name = action_names.get(action, f"UNKNOWN_{action}")
                if action_name in action_counts:
                    action_counts[action_name] += 1
                else:
                    action_counts[action_name] = 1

        # Compute total and distributions
        total_actions = sum(action_counts.values())
        action_distributions = {}

        if total_actions > 0:
            action_distributions = {
                name: (count / total_actions) * 100
                for name, count in action_counts.items()
            }
        else:
            action_distributions = {name: 0.0 for name in action_counts.keys()}

        return {
            'counts': action_counts,
            'distributions': action_distributions,
            'total': total_actions
        }

    def compute_advantages(
        self,
        trajectory: SearchTrajectory
    ) -> Tuple[List[float], List[float]]:
        """
        Compute advantages using GAE.

        Paper reward structure:
        - Intermediate rewards: R(s,a) from each action (already in trajectory.rewards)
        - Final reward: +10 if correct, -5 if incorrect (added to last step)
        - Step penalty: -0.1 per step (distributed across trajectory)

        Args:
            trajectory: Search trajectory

        Returns:
            Tuple of (advantages, returns)
        """
        rewards = trajectory.rewards.copy()
        values = [v.item() if isinstance(v, torch.Tensor) else v
                 for v in trajectory.values]

        if len(rewards) == 0:
            return [], []

        # Apply step penalty to each reward (per-step efficiency penalty)
        # This encourages the policy to find solutions with fewer steps
        for i in range(len(rewards)):
            rewards[i] += self.config.step_penalty

        # Add final reward to last step only
        if trajectory.is_correct:
            final_reward = self.config.final_reward_correct
        else:
            final_reward = self.config.final_reward_incorrect

        rewards[-1] += final_reward

        # Compute advantages using GAE
        advantages = []
        returns = []

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.discount_gamma * next_value - values[t]
            gae = delta + self.config.discount_gamma * self.config.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update_policy(
        self,
        trajectories: List[SearchTrajectory]
    ) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            trajectories: List of collected trajectories

        Returns:
            Dictionary of training metrics
        """
        # Compute advantages for all trajectories
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []

        for trajectory in trajectories:
            if len(trajectory.states) == 0:
                continue

            advantages, returns = self.compute_advantages(trajectory)

            all_states.extend(trajectory.states)
            all_actions.extend(trajectory.actions)
            all_old_log_probs.extend(trajectory.log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        if len(all_states) == 0:
            logger.warning("No valid trajectories collected")
            return {}

        # Normalize advantages
        all_advantages = torch.tensor(all_advantages, dtype=torch.float32)
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        all_actions = torch.tensor(all_actions, dtype=torch.long)
        # Detach old log probs - they represent the behavior policy and should be constants
        all_old_log_probs = torch.stack([
            lp.detach() if isinstance(lp, torch.Tensor) else torch.tensor(lp)
            for lp in all_old_log_probs
        ])
        all_returns = torch.tensor(all_returns, dtype=torch.float32)

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(all_states))

            for i in range(0, len(all_states), self.config.batch_size):
                with torch.autograd.set_detect_anomaly(True):
                    batch_indices = indices[i:i + self.config.batch_size]

                    if len(batch_indices) == 0:
                        continue

                    # Get batch data
                    batch_states = [all_states[idx] for idx in batch_indices]
                    batch_actions = all_actions[batch_indices].to(self.device)
                    batch_old_log_probs = all_old_log_probs[batch_indices].to(self.device)
                    batch_advantages = all_advantages[batch_indices].to(self.device)
                    batch_returns = all_returns[batch_indices].to(self.device)

                    # Evaluate actions with current policy
                    batch_graphs = [state['graph'].to(self.device) for state in batch_states]

                    batch_log_probs = []
                    batch_values = []
                    batch_entropies = []

                    for graph, action in zip(batch_graphs, batch_actions):
                        log_prob, value, entropy = self.policy_network.evaluate_actions(
                            graph,
                            action.unsqueeze(0)
                        )
                        batch_log_probs.append(log_prob)
                        batch_values.append(value)
                        batch_entropies.append(entropy)

                    batch_log_probs = torch.stack(batch_log_probs)
                    batch_values = torch.stack(batch_values)
                    batch_entropies = torch.stack(batch_entropies)

                    # Compute PPO loss
                    ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon
                    ) * batch_advantages

                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(batch_values, batch_returns)

                    # Entropy bonus
                    entropy_loss = -batch_entropies.mean()

                    # Total loss
                    loss = (
                        policy_loss +
                        self.config.value_coeff * value_loss +
                        self.config.entropy_coeff * entropy_loss
                    )

                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_network.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += -entropy_loss.item()
                    num_updates += 1

        # Compute metrics
        metrics = {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'num_trajectories': len(trajectories),
            'avg_trajectory_length': np.mean([len(t.states) for t in trajectories]),
            'accuracy': np.mean([t.is_correct for t in trajectories])
        }

        return metrics

    def train(
        self,
        train_problems: List[str],
        train_answers: List[str],
        num_iterations: int = 100,
        problems_per_iteration: int = 32,
        log_callback: Optional[callable] = None
    ):
        """
        Main training loop.

        Args:
            train_problems: Training problems
            train_answers: Ground truth answers
            num_iterations: Number of training iterations
            problems_per_iteration: Problems per iteration
            log_callback: Callback for logging metrics
        """
        logger.info(f"Starting PPO training for {num_iterations} iterations")

        for iteration in range(num_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # Sample problems
            indices = np.random.choice(
                len(train_problems),
                size=min(problems_per_iteration, len(train_problems)),
                replace=False
            )
            batch_problems = [train_problems[i] for i in indices]
            batch_answers = [train_answers[i] for i in indices]

            # Collect trajectories
            trajectories = self.collect_trajectories(batch_problems, batch_answers)

            # Compute action statistics
            action_stats = self.compute_action_statistics(trajectories)

            # Store action statistics
            self.action_history['iterations'].append(iteration + 1)
            self.action_history['action_counts'].append(action_stats['counts'])
            self.action_history['action_distributions'].append(action_stats['distributions'])

            # Update policy
            metrics = self.update_policy(trajectories)

            # Add action statistics to metrics
            metrics['action_counts'] = action_stats['counts']
            metrics['action_distributions'] = action_stats['distributions']

            # Log metrics
            logger.info(f"Metrics: {metrics}")
            logger.info(f"Action Statistics: {action_stats['counts']}")
            logger.info(f"Action Distribution (%): {action_stats['distributions']}")

            if log_callback:
                log_callback(iteration, metrics)

        logger.info("PPO training completed")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")

    def save_action_statistics(self, path: str):
        """
        Save action statistics to JSON file.

        Args:
            path: Path to save JSON file
        """
        import json
        with open(path, 'w') as f:
            json.dump(self.action_history, f, indent=2)
        logger.info(f"Action statistics saved to {path}")

    def get_action_history(self) -> Dict:
        """
        Get action history for visualization.

        Returns:
            Dictionary with action history
        """
        return self.action_history


# Import F for MSE loss
import torch.nn.functional as F
