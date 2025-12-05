"""
Visualization utilities for action statistics during PGTS training.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_action_distribution_over_time(
    action_history: Dict,
    output_path: str,
    title: str = "Action Distribution Over Training Iterations"
):
    """
    Plot stacked area chart showing action distribution over time.

    Args:
        action_history: Dictionary with iterations, action_counts, action_distributions
        output_path: Path to save the plot
        title: Plot title
    """
    iterations = action_history['iterations']
    action_distributions = action_history['action_distributions']

    if len(iterations) == 0:
        logger.warning("No data to plot")
        return

    # Extract action types
    action_types = list(action_distributions[0].keys())

    # Prepare data for stacked area plot
    data = {action_type: [] for action_type in action_types}

    for dist in action_distributions:
        for action_type in action_types:
            data[action_type].append(dist.get(action_type, 0.0))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot stacked area
    ax.stackplot(iterations, *[data[action_type] for action_type in action_types],
                 labels=action_types, alpha=0.8)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Action Percentage (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved stacked area plot to {output_path}")


def plot_action_counts_histogram(
    action_history: Dict,
    output_path: str,
    iteration_idx: Optional[int] = None,
    title: str = "Action Counts Histogram"
):
    """
    Plot histogram of action counts for a specific iteration or average across all.

    Args:
        action_history: Dictionary with iterations, action_counts, action_distributions
        output_path: Path to save the plot
        iteration_idx: Specific iteration index (None for average across all)
        title: Plot title
    """
    action_counts_list = action_history['action_counts']

    if len(action_counts_list) == 0:
        logger.warning("No data to plot")
        return

    if iteration_idx is not None and 0 <= iteration_idx < len(action_counts_list):
        # Single iteration
        action_counts = action_counts_list[iteration_idx]
        title = f"{title} (Iteration {action_history['iterations'][iteration_idx]})"
    else:
        # Average across all iterations
        action_types = list(action_counts_list[0].keys())
        action_counts = {action_type: 0 for action_type in action_types}

        for counts in action_counts_list:
            for action_type, count in counts.items():
                action_counts[action_type] += count

        # Average
        num_iterations = len(action_counts_list)
        action_counts = {k: v / num_iterations for k, v in action_counts.items()}
        title = f"{title} (Average Across All Iterations)"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    action_types = list(action_counts.keys())
    counts = list(action_counts.values())

    # Color palette
    colors = sns.color_palette("husl", len(action_types))

    # Plot bars
    bars = ax.bar(action_types, counts, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Action Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved histogram to {output_path}")


def plot_action_trends(
    action_history: Dict,
    output_path: str,
    title: str = "Action Trends Over Training"
):
    """
    Plot line chart showing individual action trends over iterations.

    Args:
        action_history: Dictionary with iterations, action_counts, action_distributions
        output_path: Path to save the plot
        title: Plot title
    """
    iterations = action_history['iterations']
    action_counts_list = action_history['action_counts']

    if len(iterations) == 0:
        logger.warning("No data to plot")
        return

    # Extract action types
    action_types = list(action_counts_list[0].keys())

    # Prepare data
    data = {action_type: [] for action_type in action_types}

    for counts in action_counts_list:
        for action_type in action_types:
            data[action_type].append(counts.get(action_type, 0))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color palette
    colors = sns.color_palette("husl", len(action_types))

    # Plot lines
    for idx, action_type in enumerate(action_types):
        ax.plot(iterations, data[action_type], marker='o', label=action_type,
                color=colors[idx], linewidth=2, markersize=5)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Action Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved trend plot to {output_path}")


def plot_action_heatmap(
    action_history: Dict,
    output_path: str,
    title: str = "Action Distribution Heatmap"
):
    """
    Plot heatmap showing action percentages across iterations.

    Args:
        action_history: Dictionary with iterations, action_counts, action_distributions
        output_path: Path to save the plot
        title: Plot title
    """
    iterations = action_history['iterations']
    action_distributions = action_history['action_distributions']

    if len(iterations) == 0:
        logger.warning("No data to plot")
        return

    # Extract action types
    action_types = list(action_distributions[0].keys())

    # Prepare data matrix
    data_matrix = []
    for dist in action_distributions:
        row = [dist.get(action_type, 0.0) for action_type in action_types]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix).T  # Transpose for better visualization

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot heatmap
    im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Set ticks
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels([f"Iter {i}" for i in iterations], rotation=45, ha='right')
    ax.set_yticks(range(len(action_types)))
    ax.set_yticklabels(action_types)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', fontsize=11)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Action Type', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved heatmap to {output_path}")


def plot_action_pie_charts(
    action_history: Dict,
    output_path: str,
    num_iterations: int = 4,
    title: str = "Action Distribution Snapshots"
):
    """
    Plot pie charts showing action distribution at different iterations.

    Args:
        action_history: Dictionary with iterations, action_counts, action_distributions
        output_path: Path to save the plot
        num_iterations: Number of iterations to show (evenly spaced)
        title: Plot title
    """
    iterations = action_history['iterations']
    action_distributions = action_history['action_distributions']

    if len(iterations) == 0:
        logger.warning("No data to plot")
        return

    # Select evenly spaced iterations
    total_iters = len(iterations)
    if total_iters <= num_iterations:
        selected_indices = list(range(total_iters))
    else:
        step = total_iters // num_iterations
        selected_indices = [i * step for i in range(num_iterations)]
        # Ensure last iteration is included
        if selected_indices[-1] != total_iters - 1:
            selected_indices[-1] = total_iters - 1

    # Create subplots
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(5 * len(selected_indices), 5))

    if len(selected_indices) == 1:
        axes = [axes]

    # Color palette
    action_types = list(action_distributions[0].keys())
    colors = sns.color_palette("husl", len(action_types))

    for idx, iter_idx in enumerate(selected_indices):
        ax = axes[idx]
        dist = action_distributions[iter_idx]

        # Filter out zero values
        labels = []
        sizes = []
        plot_colors = []

        for i, action_type in enumerate(action_types):
            value = dist.get(action_type, 0.0)
            if value > 0:
                labels.append(action_type)
                sizes.append(value)
                plot_colors.append(colors[i])

        # Plot pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=plot_colors,
                                           autopct='%1.1f%%', startangle=90)

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        ax.set_title(f"Iteration {iterations[iter_idx]}", fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved pie charts to {output_path}")


def generate_all_plots(
    action_history: Dict,
    output_dir: str,
    prefix: str = "action_stats"
):
    """
    Generate all action visualization plots.

    Args:
        action_history: Dictionary with action history
        output_dir: Directory to save plots
        prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating action statistics visualizations in {output_dir}")

    # 1. Stacked area plot
    plot_action_distribution_over_time(
        action_history,
        str(output_dir / f"{prefix}_distribution_over_time.png")
    )

    # 2. Histogram (average)
    plot_action_counts_histogram(
        action_history,
        str(output_dir / f"{prefix}_histogram_average.png")
    )

    # 3. Histogram (last iteration)
    if len(action_history['iterations']) > 0:
        plot_action_counts_histogram(
            action_history,
            str(output_dir / f"{prefix}_histogram_last_iteration.png"),
            iteration_idx=-1
        )

    # 4. Trend lines
    plot_action_trends(
        action_history,
        str(output_dir / f"{prefix}_trends.png")
    )

    # 5. Heatmap
    plot_action_heatmap(
        action_history,
        str(output_dir / f"{prefix}_heatmap.png")
    )

    # 6. Pie charts
    plot_action_pie_charts(
        action_history,
        str(output_dir / f"{prefix}_pie_charts.png")
    )

    logger.info("All visualizations generated successfully!")


def load_and_visualize(
    json_path: str,
    output_dir: str,
    prefix: str = "action_stats"
):
    """
    Load action statistics from JSON and generate visualizations.

    Args:
        json_path: Path to JSON file with action history
        output_dir: Directory to save plots
        prefix: Prefix for output files
    """
    logger.info(f"Loading action statistics from {json_path}")

    with open(json_path, 'r') as f:
        action_history = json.load(f)

    generate_all_plots(action_history, output_dir, prefix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize PGTS action statistics")
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to action statistics JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save plots')
    parser.add_argument('--prefix', type=str, default='action_stats',
                        help='Prefix for output files')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_and_visualize(args.json_path, args.output_dir, args.prefix)
