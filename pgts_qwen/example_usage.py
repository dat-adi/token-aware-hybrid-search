"""
Example usage of PGTS for a single problem.
"""
import torch
import logging

from models.qwen3_wrapper import Qwen3ReasoningGenerator
from models.reward_model import ProcessRewardModel
from models.policy_network import GPSPolicyNetwork
from tree_search.search_algorithm import PGTSSearch, SearchConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_problem():
    """
    Example: Solve a single GSM8k problem with PGTS.
    """
    # Example problem
    problem = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""

    ground_truth_answer = "18"

    logger.info(f"Problem: {problem}\n")

    # Initialize models (assuming pre-trained models exist)
    # For this example, we'll use random initialization
    logger.info("Initializing models...")

    # Qwen3 reasoning generator
    reasoning_generator = Qwen3ReasoningGenerator(
        model_name="Qwen/Qwen3-8B",
        temperature=0.7,
        use_vllm=False
    )

    # For demonstration, we'll skip reward model and policy network loading
    # In practice, you would load pre-trained checkpoints:
    # reward_model = ProcessRewardModel.from_pretrained("path/to/checkpoint")
    # policy_network.load_state_dict(torch.load("path/to/checkpoint.pt"))

    logger.info("Note: For a real run, load pre-trained reward model and policy network")
    logger.info("This example demonstrates the API usage\n")

    # Create a simple demo without full training
    logger.info("=== Demo: Generate reasoning steps ===\n")

    # Generate first reasoning step
    step1, hidden1 = reasoning_generator.generate_step(
        problem=problem,
        reasoning_chain=[],
        return_hidden_states=True
    )
    logger.info(f"Step 1: {step1}\n")

    # Generate second reasoning step
    step2, hidden2 = reasoning_generator.generate_step(
        problem=problem,
        reasoning_chain=[step1],
        return_hidden_states=True
    )
    logger.info(f"Step 2: {step2}\n")

    # Generate alternative branch
    alt_step, hidden_alt = reasoning_generator.generate_branch(
        problem=problem,
        reasoning_chain=[step1],
        return_hidden_states=True
    )
    logger.info(f"Alternative Step: {alt_step}\n")

    logger.info("=== Demo Complete ===")
    logger.info("\nTo run full PGTS search with trained models:")
    logger.info("1. Train reward model: python main_train.py")
    logger.info("2. Train policy network: (included in main_train.py)")
    logger.info("3. Evaluate: python main_eval.py --policy_checkpoint <path> --reward_checkpoint <path>")


def example_tree_construction():
    """
    Example: Manually construct and visualize a reasoning tree.
    """
    from tree_search.tree_state import TreeState
    import torch

    logger.info("\n=== Example: Tree Construction ===\n")

    problem = "If x + 5 = 12, what is x?"

    # Initialize tree
    tree = TreeState(root_content=problem, hidden_dim=4096)

    logger.info(f"Initial tree: {tree}")

    # Add reasoning steps
    step1_hidden = torch.randn(4096)
    tree.add_node(
        content="Subtract 5 from both sides: x + 5 - 5 = 12 - 5",
        hidden_state=step1_hidden,
        action="EXPAND",
        reward=0.9
    )

    logger.info(f"After step 1: {tree}")

    step2_hidden = torch.randn(4096)
    tree.add_node(
        content="Simplify: x = 7",
        hidden_state=step2_hidden,
        action="EXPAND",
        reward=0.95
    )

    logger.info(f"After step 2: {tree}")

    # Get solution path
    best_leaf = tree.get_best_leaf()
    solution_path = best_leaf.get_path_from_root()

    logger.info("\nSolution path:")
    for i, node in enumerate(solution_path):
        logger.info(f"  {i}. {node.content} (reward: {node.reward:.2f})")

    # Convert to graph
    graph = tree.to_graph()
    logger.info(f"\nGraph representation:")
    logger.info(f"  Nodes: {graph.num_nodes}")
    logger.info(f"  Edges: {graph.edge_index.shape[1]}")
    logger.info(f"  Node features shape: {graph.x.shape}")


def main():
    """Run examples."""
    logger.info("=" * 60)
    logger.info("PGTS Example Usage")
    logger.info("=" * 60 + "\n")

    # Example 1: Single problem (API demonstration)
    example_single_problem()

    # Example 2: Tree construction
    example_tree_construction()

    logger.info("\n" + "=" * 60)
    logger.info("Examples Complete")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
