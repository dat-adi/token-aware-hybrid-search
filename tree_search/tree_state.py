"""
Tree state representation for PGTS reasoning.
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import torch
from torch_geometric.data import Data
import numpy as np


@dataclass
class TreeNode:
    """
    Represents a single node in the reasoning tree.

    Attributes:
        node_id: Unique identifier
        content: Text content of reasoning step
        hidden_state: LLM hidden state (from last token)
        parent: Reference to parent node
        children: List of child nodes
        depth: Distance from root
        action_taken: Action that created this node
        reward: Intermediate reward from PRM
        visit_count: Number of times visited during search
    """
    node_id: int
    content: str
    hidden_state: Optional[torch.Tensor] = None
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    depth: int = 0
    action_taken: Optional[str] = None
    reward: float = 0.0
    visit_count: int = 0

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this node is the root node."""
        return self.parent is None

    def add_child(self, child: 'TreeNode') -> 'TreeNode':
        """Add a child node and set parent relationship."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        return child

    def get_path_from_root(self) -> List['TreeNode']:
        """Get the path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_reasoning_chain(self) -> str:
        """Get the reasoning chain from root to this node as text."""
        path = self.get_path_from_root()
        # Skip root (problem statement) if it's explicitly stored
        if len(path) > 1:
            return "\n".join(node.content for node in path[1:])
        return ""


class TreeState:
    """
    Complete tree structure representing search state.

    Methods:
        - to_graph(): Convert tree to PyTorch Geometric graph
        - get_current_path(): Get reasoning chain from root to current
        - add_node(): Add new node with specified action
        - backtrack(): Move to parent node
        - get_leaf_nodes(): Return all leaf nodes
        - compute_features(): Extract node and edge features
    """

    def __init__(self, root_content: str, hidden_dim: int = 4096):
        """
        Initialize tree with root node.

        Args:
            root_content: Problem statement
            hidden_dim: Dimension of hidden states
        """
        self.hidden_dim = hidden_dim
        self.root = TreeNode(
            node_id=0,
            content=root_content,
            depth=0
        )
        self.current_node = self.root
        self.nodes: Dict[int, TreeNode] = {0: self.root}
        self.next_node_id = 1

    def add_node(self, content: str, hidden_state: torch.Tensor,
                 action: str, reward: float = 0.0) -> TreeNode:
        """
        Add new node to the tree.

        Args:
            content: Reasoning step text
            hidden_state: Hidden state from LLM
            action: Action taken (EXPAND, BRANCH)
            reward: Reward from PRM

        Returns:
            The newly created node
        """
        new_node = TreeNode(
            node_id=self.next_node_id,
            content=content,
            hidden_state=hidden_state,
            action_taken=action,
            reward=reward
        )
        self.nodes[self.next_node_id] = new_node
        self.next_node_id += 1

        self.current_node.add_child(new_node)
        self.current_node = new_node

        return new_node

    def backtrack(self) -> bool:
        """
        Move to parent node.

        Returns:
            True if backtrack successful, False if at root
        """
        if self.current_node.is_root():
            return False
        self.current_node = self.current_node.parent
        return True

    def get_current_path(self) -> List[TreeNode]:
        """Get the path from root to current node."""
        return self.current_node.get_path_from_root()

    def get_leaf_nodes(self) -> List[TreeNode]:
        """Return all leaf nodes in the tree."""
        leaves = []
        for node in self.nodes.values():
            if node.is_leaf():
                leaves.append(node)
        return leaves

    def get_all_nodes(self) -> List[TreeNode]:
        """Return all nodes in the tree."""
        return list(self.nodes.values())

    def to_graph(self) -> Data:
        """
        Convert tree to PyTorch Geometric graph.

        Returns:
            PyG Data object with node features, edge indices, and edge features
        """
        num_nodes = len(self.nodes)

        # Initialize node features (hidden states)
        node_features = torch.zeros(num_nodes, self.hidden_dim)
        for node in self.nodes.values():
            if node.hidden_state is not None:
                node_features[node.node_id] = node.hidden_state

        # Build edge index and edge features
        edge_indices = []
        edge_features = []

        for node in self.nodes.values():
            if not node.is_root():
                parent_id = node.parent.node_id
                child_id = node.node_id

                # Add edge from parent to child
                edge_indices.append([parent_id, child_id])

                # Edge features: [step_reward, depth_delta]
                depth_delta = node.depth - node.parent.depth
                edge_features.append([node.reward, depth_delta])

        # Convert to tensors
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Single node (root only)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)

        # Create node-level attributes
        node_rewards = torch.tensor([node.reward for node in self.nodes.values()],
                                    dtype=torch.float)
        node_depths = torch.tensor([node.depth for node in self.nodes.values()],
                                   dtype=torch.long)

        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_rewards=node_rewards,
            node_depths=node_depths,
            num_nodes=num_nodes,
            current_node_idx=self.current_node.node_id
        )

        return data

    def compute_features(self) -> Dict[str, Any]:
        """
        Extract features from the current tree state.

        Returns:
            Dictionary of tree features
        """
        nodes = self.get_all_nodes()
        leaves = self.get_leaf_nodes()

        return {
            'num_nodes': len(nodes),
            'num_leaves': len(leaves),
            'max_depth': max(node.depth for node in nodes),
            'current_depth': self.current_node.depth,
            'avg_reward': np.mean([node.reward for node in nodes if node.reward != 0.0])
                         if any(node.reward != 0.0 for node in nodes) else 0.0,
            'path_length': len(self.get_current_path()) - 1  # Exclude root
        }

    def get_best_leaf(self) -> TreeNode:
        """
        Get the leaf node with the highest cumulative reward.

        Returns:
            Best leaf node
        """
        leaves = self.get_leaf_nodes()
        if not leaves:
            return self.root

        best_leaf = max(leaves, key=lambda node: sum(
            n.reward for n in node.get_path_from_root()
        ))
        return best_leaf

    def __repr__(self) -> str:
        features = self.compute_features()
        return f"TreeState(nodes={features['num_nodes']}, depth={features['max_depth']}, current_depth={features['current_depth']})"
