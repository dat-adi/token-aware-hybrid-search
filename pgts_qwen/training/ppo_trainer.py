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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import F for MSE loss
import torch.nn.functional as F

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
        device: str = "cuda",
        log_file: Optional[str] = None
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_network: GPS policy network
            reasoning_generator: Qwen3 model
            reward_model: Process reward model
            config: PPO configuration
            device: Device to use
            log_file: Path to detailed log file
        """
        self.policy_network = policy_network.to(device)
        self.reasoning_generator = reasoning_generator
        self.reward_model = reward_model
        self.config = config or PPOConfig()
        self.device = device
        self.log_file = log_file
        self.iteration_count = 0

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

        # Initialize log file
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("PGTS Training Log\n")
                f.write("="*80 + "\n\n")

        logger.info("PPO Trainer initialized")

    def _log_tree_visualization(self, tree, file, problem_idx):
        """Log tree visualization to file."""
        from tree_search.tree_state import TreeNode

        file.write("\n" + "="*80 + "\n")
        file.write(f"Tree Visualization - Problem {problem_idx}\n")
        file.write("="*80 + "\n")

        def truncate(text: str, max_len: int = 60) -> str:
            text = text.replace('\n', ' ').strip()
            if len(text) > max_len:
                return text[:max_len-3] + "..."
            return text

        def visit_node(node: TreeNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            content_preview = truncate(node.content, 60)
            node_info = f"Node{node.node_id}"

            if node.is_root():
                node_info += " [ROOT]"
            else:
                node_info += f" [d={node.depth}, r={node.reward:.2f}]"

            if node == tree.current_node:
                node_info += " ← CURRENT"

            file.write(f"{prefix}{connector}{node_info}\n")
            file.write(f"{prefix}{'    ' if is_last else '│   '}  {content_preview}\n")

            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                visit_node(child, child_prefix, is_last_child)

        visit_node(tree.root)

        features = tree.compute_features()
        file.write("\n" + "-"*80 + "\n")
        file.write("Tree Statistics:\n")
        file.write(f"  Total Nodes: {features['num_nodes']}\n")
        file.write(f"  Leaf Nodes: {features['num_leaves']}\n")
        file.write(f"  Max Depth: {features['max_depth']}\n")
        file.write(f"  Current Depth: {features['current_depth']}\n")
        file.write(f"  Avg Reward: {features['avg_reward']:.3f}\n")
        file.write("="*80 + "\n\n")

        # Best reasoning path
        best_leaf = tree.get_best_leaf()
        path = best_leaf.get_path_from_root()

        file.write("Best Reasoning Path:\n")
        file.write("="*80 + "\n")
        cumulative_reward = 0.0

        for i, node in enumerate(path):
            if node.is_root():
                file.write(f"\n[PROBLEM]\n{node.content[:200]}...\n")
            else:
                cumulative_reward += node.reward
                file.write(f"\n[STEP {i}] (reward={node.reward:.3f}, cumulative={cumulative_reward:.3f})\n")
                file.write(f"{node.content[:300]}...\n")

        file.write(f"\nFinal Cumulative Reward: {cumulative_reward:.3f}\n")
        file.write("="*80 + "\n\n")

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

                # Detailed logging for debugging
                if self.log_file:
                    with open(self.log_file, 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Iteration {self.iteration_count} - Problem {idx + 1}\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"Question: {problem[:200]}...\n")
                        f.write(f"Ground Truth: {ground_truths[idx]}\n")
                        f.write(f"Predicted Answer: {trajectory.final_answer}\n")
                        f.write(f"Is Correct: {trajectory.is_correct}\n")
                        f.write(f"Num Steps: {len(trajectory.states)}\n")
                        f.write(f"Total Reward: {sum(trajectory.rewards):.3f}\n")

                        # Log reasoning path
                        best_leaf = tree.get_best_leaf()
                        reasoning_path = best_leaf.get_path_from_root()[1:]  # Exclude root
                        f.write(f"\nReasoning Path ({len(reasoning_path)} steps):\n")
                        for i, node in enumerate(reasoning_path):
                            f.write(f"\nStep {i+1} (reward={node.reward:.3f}):\n")
                            f.write(f"{node.content[:300]}...\n")

                        # Visualize tree for first problem and periodically
                        if idx == 0 or (idx + 1) % 10 == 0:
                            self._log_tree_visualization(tree, f, idx + 1)

            trajectories.append(trajectory)

        return trajectories

    def compute_advantages(
        self,
        trajectory: SearchTrajectory
    ) -> Tuple[List[float], List[float]]:
        """
        Compute advantages using GAE.

        Args:
            trajectory: Search trajectory

        Returns:
            Tuple of (advantages, returns)
        """
        rewards = trajectory.rewards.copy()
        values = [v.item() if isinstance(v, torch.Tensor) else v
                 for v in trajectory.values]

        # Add final reward
        if trajectory.is_correct:
            final_reward = self.config.final_reward_correct
        else:
            final_reward = self.config.final_reward_incorrect

        # Add step penalties
        num_steps = len(rewards)
        step_penalty = self.config.step_penalty * num_steps

        # Adjust last reward
        if len(rewards) > 0:
            rewards[-1] += final_reward + step_penalty

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
        correct_count = sum([t.is_correct for t in trajectories])
        total_count = len(trajectories)

        metrics = {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'num_trajectories': len(trajectories),
            'avg_trajectory_length': np.mean([len(t.states) for t in trajectories]),
            'accuracy': np.mean([t.is_correct for t in trajectories]),
            'correct_count': correct_count,
            'total_count': total_count,
            'avg_reward': np.mean([sum(t.rewards) for t in trajectories]),
            'num_with_answers': sum([t.final_answer is not None for t in trajectories])
        }

        # Log detailed metrics
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Update Metrics\n")
                f.write(f"{'='*80}\n")
                f.write(f"Policy Loss: {metrics['policy_loss']:.4f}\n")
                f.write(f"Value Loss: {metrics['value_loss']:.4f}\n")
                f.write(f"Entropy: {metrics['entropy']:.4f}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f} ({correct_count}/{total_count})\n")
                f.write(f"Avg Trajectory Length: {metrics['avg_trajectory_length']:.2f}\n")
                f.write(f"Avg Reward: {metrics['avg_reward']:.3f}\n")
                f.write(f"Problems with Answers: {metrics['num_with_answers']}/{total_count}\n")
                f.write("\n")

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
            self.iteration_count = iteration
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

            # Update policy
            metrics = self.update_policy(trajectories)

            # Log metrics
            logger.info(f"Metrics: {metrics}")

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



