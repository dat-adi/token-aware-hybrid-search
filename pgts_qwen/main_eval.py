"""
Main evaluation script for PGTS.
"""
import torch
import yaml
import logging
import argparse
import json
import os
import time
from pathlib import Path
from tqdm import tqdm

from models.qwen3_wrapper import Qwen3ReasoningGenerator
from models.reward_model import ProcessRewardModel
from models.policy_network import GPSPolicyNetwork
from data.gsm8k_loader import load_gsm8k
from tree_search.search_algorithm import PGTSSearch, SearchConfig
from evaluation.metrics import PGTSMetrics, compare_with_baselines

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_pgts(
    policy_checkpoint: str,
    reward_checkpoint: str,
    model_config: dict,
    eval_config: dict,
    test_set: list,
    output_dir: str
):
    """
    Evaluate PGTS on GSM8k test set.

    Args:
        policy_checkpoint: Path to policy checkpoint
        reward_checkpoint: Path to reward model checkpoint
        model_config: Model configuration
        eval_config: Evaluation configuration
        test_set: Test data
        output_dir: Output directory

    Returns:
        Evaluation results
    """
    logger.info("=== Evaluating PGTS ===")

    # Load models
    logger.info("Loading models...")
    reasoning_generator = Qwen3ReasoningGenerator(
        model_name=model_config['qwen3']['model_name'],
        temperature=model_config['qwen3']['temperature'],
        use_vllm=False
    )

    reward_model = ProcessRewardModel.from_pretrained(reward_checkpoint)

    policy_network = GPSPolicyNetwork(
        input_dim=reasoning_generator.get_hidden_dim(),
        hidden_dim=model_config['policy_network']['hidden_dim'],
        num_layers=model_config['policy_network']['num_layers'],
        num_heads=model_config['policy_network']['num_heads'],
        dropout=model_config['policy_network']['dropout'],
        activation=model_config['policy_network']['activation']
    )

    # Load policy checkpoint
    checkpoint = torch.load(policy_checkpoint, map_location='cuda')
    policy_network.load_state_dict(checkpoint['policy_state_dict'])
    policy_network.eval()

    # Initialize search algorithm
    search_config = SearchConfig(
        max_depth=20,
        max_nodes=100,
        temperature=0.8,
        use_policy=True
    )

    search_algorithm = PGTSSearch(
        reasoning_generator=reasoning_generator,
        reward_model=reward_model,
        policy_network=policy_network,
        config=search_config
    )

    # Initialize metrics
    metrics = PGTSMetrics()

    # Evaluate on test set
    results = []
    num_samples = min(len(test_set), eval_config['evaluation']['num_samples'])

    logger.info(f"Evaluating on {num_samples} test examples...")

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        example = test_set[idx]
        problem = example['problem']
        ground_truth = example['answer']

        # Run search
        start_time = time.time()
        tree, trajectory = search_algorithm.search(problem, collect_trajectory=True)
        search_time = time.time() - start_time

        # Evaluate solution
        is_correct = search_algorithm.evaluate_solution(trajectory, ground_truth)

        # Get best leaf
        best_leaf = tree.get_best_leaf()
        tree_features = tree.compute_features()

        # Update metrics
        metrics.update(
            is_correct=is_correct,
            num_nodes=tree_features['num_nodes'],
            depth=best_leaf.depth,
            search_time=search_time,
            has_solution=trajectory.final_answer is not None
        )

        # Store result
        result = {
            'problem': problem,
            'ground_truth': ground_truth,
            'predicted_answer': trajectory.final_answer,
            'is_correct': is_correct,
            'num_nodes': tree_features['num_nodes'],
            'depth': best_leaf.depth,
            'search_time': search_time
        }
        results.append(result)

        # Save trajectory if requested
        if eval_config['evaluation'].get('save_trajectories', False):
            trajectory_dir = os.path.join(output_dir, 'trajectories')
            os.makedirs(trajectory_dir, exist_ok=True)

            trajectory_file = os.path.join(trajectory_dir, f'trajectory_{idx}.json')
            with open(trajectory_file, 'w') as f:
                json.dump({
                    'problem': problem,
                    'actions': trajectory.actions,
                    'rewards': trajectory.rewards,
                    'final_answer': trajectory.final_answer,
                    'is_correct': is_correct
                }, f, indent=2)

    # Compute final metrics
    final_metrics = metrics.compute()

    logger.info("\n" + str(metrics))

    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

    return final_metrics, results


def evaluate_baseline_cot(
    model_config: dict,
    test_set: list,
    num_samples: int = None
):
    """
    Evaluate baseline Chain-of-Thought.

    Args:
        model_config: Model configuration
        test_set: Test data
        num_samples: Number of samples to evaluate

    Returns:
        Baseline metrics
    """
    logger.info("Evaluating baseline CoT...")

    reasoning_generator = Qwen3ReasoningGenerator(
        model_name=model_config['qwen3']['model_name'],
        temperature=0.7,
        use_vllm=False
    )

    metrics = PGTSMetrics()

    num_samples = num_samples or len(test_set)

    for idx in tqdm(range(num_samples), desc="CoT Baseline"):
        example = test_set[idx]
        problem = example['problem']
        ground_truth = example['answer']

        # Generate CoT solution
        prompt = f"""Problem: {problem}

Let's solve this step by step:

Step 1:"""

        generated_text, _ = reasoning_generator._generate(prompt, return_hidden_states=False)

        # Extract answer
        predicted_answer = reasoning_generator.extract_answer(generated_text) if hasattr(reasoning_generator, 'extract_answer') else None

        # Evaluate
        is_correct = False
        if predicted_answer:
            try:
                pred_num = float(predicted_answer.replace(",", ""))
                gt_num = float(ground_truth.replace(",", ""))
                is_correct = abs(pred_num - gt_num) < 1e-6
            except:
                is_correct = predicted_answer.strip() == ground_truth.strip()

        metrics.update(
            is_correct=is_correct,
            num_nodes=1,  # CoT uses 1 "node"
            depth=1,
            has_solution=predicted_answer is not None
        )

    return metrics.compute()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate PGTS on GSM8k')
    parser.add_argument('--policy_checkpoint', type=str, required=True,
                       help='Path to policy checkpoint')
    parser.add_argument('--reward_checkpoint', type=str, required=True,
                       help='Path to reward model checkpoint')
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml',
                       help='Path to model config')
    parser.add_argument('--eval_config', type=str, default='config/eval_config.yaml',
                       help='Path to eval config')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory')
    parser.add_argument('--compare_baselines', action='store_true',
                       help='Compare with baseline methods')

    args = parser.parse_args()

    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_config(args.model_config)
    eval_config = load_config(args.eval_config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    logger.info("Loading GSM8k test set...")
    _, _, test_data = load_gsm8k()

    # Evaluate PGTS
    pgts_metrics, results = evaluate_pgts(
        policy_checkpoint=args.policy_checkpoint,
        reward_checkpoint=args.reward_checkpoint,
        model_config=model_config,
        eval_config=eval_config,
        test_set=test_data,
        output_dir=args.output_dir
    )

    # Compare with baselines
    if args.compare_baselines:
        logger.info("\n=== Evaluating Baselines ===")

        cot_metrics = evaluate_baseline_cot(
            model_config=model_config,
            test_set=test_data,
            num_samples=eval_config['evaluation']['num_samples']
        )

        baseline_results = {
            'CoT': cot_metrics
        }

        comparison = compare_with_baselines(pgts_metrics, baseline_results)
        logger.info(comparison)

        # Save comparison
        comparison_file = os.path.join(args.output_dir, 'comparison.txt')
        with open(comparison_file, 'w') as f:
            f.write(comparison)

    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
