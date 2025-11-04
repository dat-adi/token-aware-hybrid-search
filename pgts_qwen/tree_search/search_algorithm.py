"""
PGTS search algorithm implementation.
"""
import torch
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

from ..tree_search.tree_state import TreeState, TreeNode
from ..models.qwen3_wrapper import Qwen3ReasoningGenerator
from ..models.reward_model import ProcessRewardModel
from ..models.policy_network import GPSPolicyNetwork, create_action_mask

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for PGTS search."""
    max_depth: int = 20
    max_nodes: int = 100
    temperature: float = 0.8
    use_policy: bool = True
    device: str = "cuda"


@dataclass
class SearchTrajectory:
    """Trajectory data for training."""
    states: List[Dict]  # Graph representations
    actions: List[int]
    rewards: List[float]
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    problem: str
    final_answer: Optional[str] = None
    is_correct: bool = False


class PGTSSearch:
    """
    Main PGTS search algorithm.

    Parameters:
        - max_depth: 20 (maximum reasoning steps)
        - max_nodes: 100 (budget constraint)
        - temperature: 0.8 (for action sampling)

    Algorithm:
        1. Initialize tree with root (problem statement)
        2. While not terminated and budget remaining:
            a. Convert tree to graph representation
            b. Get action from policy network
            c. Execute action (expand/branch/backtrack/terminate)
            d. Compute reward for new state
            e. Record trajectory for training
        3. Return best solution path

    Methods:
        - search(): Execute full search
        - execute_action(): Apply action to tree
        - evaluate_solution(): Check final answer
        - collect_trajectory(): Store (state, action, reward) tuples
    """

    # Action definitions
    ACTION_EXPAND = 0
    ACTION_BRANCH = 1
    ACTION_BACKTRACK = 2
    ACTION_TERMINATE = 3

    ACTION_NAMES = {
        0: "EXPAND",
        1: "BRANCH",
        2: "BACKTRACK",
        3: "TERMINATE"
    }

    def __init__(
        self,
        reasoning_generator: Qwen3ReasoningGenerator,
        reward_model: ProcessRewardModel,
        policy_network: Optional[GPSPolicyNetwork] = None,
        config: Optional[SearchConfig] = None
    ):
        """
        Initialize PGTS search.

        Args:
            reasoning_generator: Qwen3 model for generating steps
            reward_model: Process reward model
            policy_network: Policy network (optional, uses random if None)
            config: Search configuration
        """
        self.reasoning_generator = reasoning_generator
        self.reward_model = reward_model
        self.policy_network = policy_network
        self.config = config or SearchConfig()

        self.hidden_dim = reasoning_generator.get_hidden_dim()

        logger.info(f"PGTS Search initialized with max_depth={self.config.max_depth}, max_nodes={self.config.max_nodes}")

    def search(
        self,
        problem: str,
        collect_trajectory: bool = False
    ) -> Tuple[TreeState, Optional[SearchTrajectory]]:
        """
        Execute full PGTS search.

        Args:
            problem: Math problem to solve
            collect_trajectory: Whether to collect trajectory for training

        Returns:
            Tuple of (final_tree_state, trajectory)
        """
        # Initialize tree
        tree = TreeState(root_content=problem, hidden_dim=self.hidden_dim)

        # Initialize trajectory
        trajectory = SearchTrajectory(
            states=[],
            actions=[],
            rewards=[],
            log_probs=[],
            values=[],
            problem=problem
        ) if collect_trajectory else None

        # Search loop
        num_nodes = 1  # Start with root
        terminated = False

        while not terminated and num_nodes < self.config.max_nodes:
            # Convert tree to graph
            graph = tree.to_graph()

            # Add action mask
            graph.action_mask = create_action_mask(tree)
            graph = graph.to(self.config.device)

            # Select action
            if self.policy_network is not None and self.config.use_policy:
                action, log_prob, value = self.policy_network.select_action(
                    graph,
                    temperature=self.config.temperature,
                    deterministic=False
                )
            else:
                # Random policy
                valid_actions = torch.where(graph.action_mask)[0]
                action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
                log_prob = torch.tensor(0.0)
                value = torch.tensor(0.0)

            logger.debug(f"Step {num_nodes}: Action={self.ACTION_NAMES[action]}, Depth={tree.current_node.depth}")

            # Store state before action
            if collect_trajectory:
                trajectory.states.append({
                    'graph': graph.cpu(),
                    'tree_features': tree.compute_features()
                })
                trajectory.actions.append(action)
                trajectory.log_probs.append(log_prob)
                trajectory.values.append(value)

            # Execute action
            success, reward = self.execute_action(tree, action, problem)

            if collect_trajectory:
                trajectory.rewards.append(reward)

            # Check termination
            if action == self.ACTION_TERMINATE:
                terminated = True
                logger.info(f"Search terminated at {num_nodes} nodes")
            elif not success:
                # If action failed, terminate
                logger.warning(f"Action {self.ACTION_NAMES[action]} failed, terminating")
                terminated = True

            # Update node count
            num_nodes = len(tree.nodes)

        # Get final solution
        best_leaf = tree.get_best_leaf()
        final_path = best_leaf.get_path_from_root()[1:]  # Exclude root

        if collect_trajectory:
            if len(final_path) > 0:
                trajectory.final_answer = self.extract_answer(final_path[-1].content)
            else:
                trajectory.final_answer = None

        logger.info(f"Search completed: {num_nodes} nodes, depth={best_leaf.depth}")

        return tree, trajectory

    def execute_action(
        self,
        tree: TreeState,
        action: int,
        problem: str
    ) -> Tuple[bool, float]:
        """
        Execute action on tree.

        Args:
            tree: Current tree state
            action: Action to execute
            problem: Original problem

        Returns:
            Tuple of (success, reward)
        """
        if action == self.ACTION_EXPAND:
            return self.action_expand(tree, problem)
        elif action == self.ACTION_BRANCH:
            return self.action_branch(tree, problem)
        elif action == self.ACTION_BACKTRACK:
            return self.action_backtrack(tree)
        elif action == self.ACTION_TERMINATE:
            return True, 0.0
        else:
            raise ValueError(f"Unknown action: {action}")

    def action_expand(
        self,
        tree: TreeState,
        problem: str
    ) -> Tuple[bool, float]:
        """
        EXPAND: Generate next reasoning step.

        Args:
            tree: Current tree state
            problem: Original problem

        Returns:
            Tuple of (success, reward)
        """
        # Get current reasoning chain
        current_path = tree.get_current_path()[1:]  # Exclude root
        reasoning_chain = [node.content for node in current_path]

        # Generate next step
        try:
            generated_text, hidden_state = self.reasoning_generator.generate_step(
                problem,
                reasoning_chain,
                return_hidden_states=True
            )

            if len(generated_text) == 0:
                logger.warning("Empty generation, action failed")
                return False, -1.0

            # If hidden state is None (e.g., vLLM), extract it
            if hidden_state is None:
                full_text = problem + "\n" + "\n".join(reasoning_chain) + "\n" + generated_text
                hidden_state = self.reasoning_generator.extract_hidden_states(full_text)

            # Compute reward
            new_chain = reasoning_chain + [generated_text]
            reward = self.reward_model.compute_step_reward(problem, new_chain, len(new_chain) - 1)

            # Add node to tree
            tree.add_node(
                content=generated_text,
                hidden_state=hidden_state,
                action="EXPAND",
                reward=reward
            )

            return True, reward

        except Exception as e:
            logger.error(f"Error in EXPAND action: {e}")
            return False, -1.0

    def action_branch(
        self,
        tree: TreeState,
        problem: str
    ) -> Tuple[bool, float]:
        """
        BRANCH: Explore alternative reasoning path.

        Args:
            tree: Current tree state
            problem: Original problem

        Returns:
            Tuple of (success, reward)
        """
        # Get current reasoning chain
        current_path = tree.get_current_path()[1:]  # Exclude root
        reasoning_chain = [node.content for node in current_path]

        # Generate alternative step
        try:
            generated_text, hidden_state = self.reasoning_generator.generate_branch(
                problem,
                reasoning_chain,
                return_hidden_states=True
            )

            if len(generated_text) == 0:
                logger.warning("Empty generation, action failed")
                return False, -1.0

            # If hidden state is None, extract it
            if hidden_state is None:
                full_text = problem + "\n" + "\n".join(reasoning_chain) + "\n" + generated_text
                hidden_state = self.reasoning_generator.extract_hidden_states(full_text)

            # Compute reward
            new_chain = reasoning_chain + [generated_text]
            reward = self.reward_model.compute_step_reward(problem, new_chain, len(new_chain) - 1)

            # Add node to tree
            tree.add_node(
                content=generated_text,
                hidden_state=hidden_state,
                action="BRANCH",
                reward=reward
            )

            return True, reward

        except Exception as e:
            logger.error(f"Error in BRANCH action: {e}")
            return False, -1.0

    def action_backtrack(
        self,
        tree: TreeState
    ) -> Tuple[bool, float]:
        """
        BACKTRACK: Return to parent node.

        Args:
            tree: Current tree state

        Returns:
            Tuple of (success, reward)
        """
        success = tree.backtrack()
        return success, 0.0  # No reward for backtracking

    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract final answer from reasoning step.

        Args:
            text: Reasoning step text

        Returns:
            Extracted answer or None
        """
        # Look for answer markers
        if "####" in text:
            answer = text.split("####")[-1].strip()
            return answer
        elif "answer is" in text.lower():
            parts = text.lower().split("answer is")
            if len(parts) > 1:
                answer = parts[-1].strip().split()[0]
                return answer

        return None

    def evaluate_solution(
        self,
        trajectory: SearchTrajectory,
        ground_truth: str
    ) -> bool:
        """
        Evaluate if solution is correct.

        Args:
            trajectory: Search trajectory
            ground_truth: Ground truth answer

        Returns:
            True if correct
        """
        if trajectory.final_answer is None:
            return False

        # Normalize answers for comparison
        try:
            pred_num = float(trajectory.final_answer.replace(",", ""))
            gt_num = float(ground_truth.replace(",", ""))
            return abs(pred_num - gt_num) < 1e-6
        except:
            # String comparison
            return trajectory.final_answer.strip() == ground_truth.strip()
