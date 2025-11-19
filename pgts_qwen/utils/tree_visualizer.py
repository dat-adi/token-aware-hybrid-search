"""
Tree visualization utilities for PGTS.
"""
from typing import Optional, TextIO
from tree_search.tree_state import TreeState, TreeNode


def visualize_tree_ascii(tree: TreeState, max_content_len: int = 60) -> str:
    """
    Create ASCII visualization of the reasoning tree.

    Args:
        tree: TreeState object
        max_content_len: Maximum length of content to display

    Returns:
        ASCII string representation
    """
    lines = []
    lines.append("\n" + "="*80)
    lines.append("Tree Visualization")
    lines.append("="*80)

    def truncate(text: str, max_len: int) -> str:
        """Truncate text with ellipsis."""
        text = text.replace('\n', ' ').strip()
        if len(text) > max_len:
            return text[:max_len-3] + "..."
        return text

    def visit_node(node: TreeNode, prefix: str = "", is_last: bool = True):
        """Recursively visit nodes and build ASCII tree."""
        # Node marker
        connector = "└── " if is_last else "├── "

        # Node info
        content_preview = truncate(node.content, max_content_len)
        node_info = f"Node{node.node_id}"

        if node.is_root():
            node_info += " [ROOT]"
        else:
            node_info += f" [d={node.depth}, r={node.reward:.2f}]"

        # Current node marker
        if node == tree.current_node:
            node_info += " ← CURRENT"

        lines.append(f"{prefix}{connector}{node_info}")
        lines.append(f"{prefix}{'    ' if is_last else '│   '}  {content_preview}")

        # Visit children
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            visit_node(child, child_prefix, is_last_child)

    # Start from root
    visit_node(tree.root)

    # Add statistics
    features = tree.compute_features()
    lines.append("\n" + "-"*80)
    lines.append("Tree Statistics:")
    lines.append(f"  Total Nodes: {features['num_nodes']}")
    lines.append(f"  Leaf Nodes: {features['num_leaves']}")
    lines.append(f"  Max Depth: {features['max_depth']}")
    lines.append(f"  Current Depth: {features['current_depth']}")
    lines.append(f"  Avg Reward: {features['avg_reward']:.3f}")
    lines.append("="*80 + "\n")

    return "\n".join(lines)


def visualize_reasoning_path(tree: TreeState, leaf_node: Optional[TreeNode] = None) -> str:
    """
    Visualize the reasoning path from root to a leaf node.

    Args:
        tree: TreeState object
        leaf_node: Target leaf node (uses best leaf if None)

    Returns:
        Formatted reasoning path
    """
    if leaf_node is None:
        leaf_node = tree.get_best_leaf()

    path = leaf_node.get_path_from_root()

    lines = []
    lines.append("\n" + "="*80)
    lines.append(f"Best Reasoning Path (Depth {leaf_node.depth})")
    lines.append("="*80)

    cumulative_reward = 0.0

    for i, node in enumerate(path):
        if node.is_root():
            lines.append(f"\n[PROBLEM]")
            lines.append(node.content)
        else:
            cumulative_reward += node.reward
            lines.append(f"\n[STEP {i}] (reward={node.reward:.3f}, cumulative={cumulative_reward:.3f})")
            lines.append(node.content)

    lines.append("\n" + "="*80)
    lines.append(f"Final Cumulative Reward: {cumulative_reward:.3f}")
    lines.append("="*80 + "\n")

    return "\n".join(lines)


def log_tree_visualization(
    tree: TreeState,
    file: TextIO,
    iteration: int,
    problem_idx: int,
    include_full_tree: bool = True,
    include_best_path: bool = True
):
    """
    Log tree visualization to a file.

    Args:
        tree: TreeState object
        file: File handle to write to
        iteration: Training iteration number
        problem_idx: Problem index in batch
        include_full_tree: Whether to include full tree ASCII viz
        include_best_path: Whether to include best reasoning path
    """
    file.write("\n" + "="*80 + "\n")
    file.write(f"Iteration {iteration} - Problem {problem_idx}\n")
    file.write("="*80 + "\n")

    if include_full_tree:
        file.write(visualize_tree_ascii(tree))
        file.write("\n")

    if include_best_path:
        file.write(visualize_reasoning_path(tree))
        file.write("\n")


def visualize_tree_compact(tree: TreeState) -> str:
    """
    Create compact single-line tree visualization.

    Args:
        tree: TreeState object

    Returns:
        Compact string representation
    """
    features = tree.compute_features()
    return (f"Tree[nodes={features['num_nodes']}, "
            f"depth={features['max_depth']}, "
            f"leaves={features['num_leaves']}, "
            f"avg_reward={features['avg_reward']:.2f}]")
