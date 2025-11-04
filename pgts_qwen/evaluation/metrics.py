"""
Evaluation metrics for PGTS.
"""
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PGTSMetrics:
    """
    Metrics for evaluating PGTS performance.

    Metrics:
        - accuracy: Percentage of correct final answers
        - avg_nodes: Average nodes explored per problem
        - avg_depth: Average depth of solution paths
        - efficiency: Accuracy / avg_nodes (higher is better)
        - solve_rate: Percentage of problems reaching solution
        - token_usage: Total tokens generated
        - search_time: Average inference time per problem
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.correct_count = 0
        self.total_count = 0
        self.total_nodes = 0
        self.total_depth = 0
        self.total_tokens = 0
        self.total_time = 0
        self.solved_count = 0

    def update(
        self,
        is_correct: bool,
        num_nodes: int,
        depth: int,
        num_tokens: int = 0,
        search_time: float = 0.0,
        has_solution: bool = True
    ):
        """
        Update metrics with results from one problem.

        Args:
            is_correct: Whether answer is correct
            num_nodes: Number of nodes explored
            depth: Depth of solution
            num_tokens: Tokens generated
            search_time: Search time in seconds
            has_solution: Whether solution was found
        """
        self.total_count += 1

        if is_correct:
            self.correct_count += 1

        if has_solution:
            self.solved_count += 1

        self.total_nodes += num_nodes
        self.total_depth += depth
        self.total_tokens += num_tokens
        self.total_time += search_time

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary of metrics
        """
        if self.total_count == 0:
            return {}

        accuracy = self.correct_count / self.total_count
        avg_nodes = self.total_nodes / self.total_count
        avg_depth = self.total_depth / self.total_count
        efficiency = accuracy / avg_nodes if avg_nodes > 0 else 0
        solve_rate = self.solved_count / self.total_count
        avg_tokens = self.total_tokens / self.total_count
        avg_time = self.total_time / self.total_count

        return {
            'accuracy': accuracy * 100,
            'avg_nodes': avg_nodes,
            'avg_depth': avg_depth,
            'efficiency': efficiency,
            'solve_rate': solve_rate * 100,
            'avg_tokens': avg_tokens,
            'avg_time': avg_time
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.compute()
        if not metrics:
            return "No metrics available"

        return f"""
PGTS Metrics:
  Accuracy:     {metrics['accuracy']:.2f}%
  Solve Rate:   {metrics['solve_rate']:.2f}%
  Avg Nodes:    {metrics['avg_nodes']:.1f}
  Avg Depth:    {metrics['avg_depth']:.1f}
  Efficiency:   {metrics['efficiency']:.4f}
  Avg Tokens:   {metrics['avg_tokens']:.0f}
  Avg Time:     {metrics['avg_time']:.2f}s
"""


def compare_with_baselines(
    pgts_results: Dict[str, float],
    baseline_results: Dict[str, Dict[str, float]]
) -> str:
    """
    Compare PGTS results with baselines.

    Args:
        pgts_results: PGTS metrics
        baseline_results: Dict of baseline_name -> metrics

    Returns:
        Formatted comparison string
    """
    comparison = "\n=== Baseline Comparison ===\n\n"
    comparison += f"{'Method':<20} {'Accuracy':<12} {'Avg Nodes':<12} {'Efficiency':<12}\n"
    comparison += "-" * 60 + "\n"

    # Add baselines
    for name, metrics in baseline_results.items():
        comparison += f"{name:<20} {metrics.get('accuracy', 0):<12.2f} "
        comparison += f"{metrics.get('avg_nodes', 0):<12.1f} "
        comparison += f"{metrics.get('efficiency', 0):<12.4f}\n"

    # Add PGTS
    comparison += f"{'PGTS (Ours)':<20} {pgts_results.get('accuracy', 0):<12.2f} "
    comparison += f"{pgts_results.get('avg_nodes', 0):<12.1f} "
    comparison += f"{pgts_results.get('efficiency', 0):<12.4f}\n"

    # Compute improvements
    if 'CoT' in baseline_results:
        cot_acc = baseline_results['CoT'].get('accuracy', 0)
        if cot_acc > 0:
            improvement = pgts_results.get('accuracy', 0) - cot_acc
            comparison += f"\nImprovement over CoT: +{improvement:.2f}%\n"

    return comparison


def compute_aggregate_stats(results: List[Dict]) -> Dict[str, any]:
    """
    Compute aggregate statistics from results.

    Args:
        results: List of per-problem results

    Returns:
        Aggregate statistics
    """
    if not results:
        return {}

    accuracies = [r['is_correct'] for r in results]
    nodes = [r['num_nodes'] for r in results]
    depths = [r['depth'] for r in results]

    stats = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_nodes': np.mean(nodes),
        'std_nodes': np.std(nodes),
        'median_nodes': np.median(nodes),
        'mean_depth': np.mean(depths),
        'std_depth': np.std(depths),
        'median_depth': np.median(depths),
        'min_nodes': np.min(nodes),
        'max_nodes': np.max(nodes),
        'percentile_25_nodes': np.percentile(nodes, 25),
        'percentile_75_nodes': np.percentile(nodes, 75)
    }

    return stats
