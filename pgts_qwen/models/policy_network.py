"""
Graph Policy Network using GPS architecture for action selection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_random_walk_pe(edge_index, num_nodes, k_steps=8):
    """
    Compute Random Walk Positional Encodings (RWSE).

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes in graph
        k_steps: Number of random walk steps

    Returns:
        RWSE features [num_nodes, k_steps]
    """
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0

    # Compute degree matrix for normalization
    deg = adj.sum(dim=1, keepdim=True)
    deg = torch.where(deg > 0, deg, torch.ones_like(deg))  # Avoid division by zero

    # Transition matrix: P = D^-1 * A
    trans_matrix = adj / deg

    # Compute k-step random walk probabilities
    rwse = torch.zeros(num_nodes, k_steps, device=edge_index.device)
    current = torch.eye(num_nodes, device=edge_index.device)

    for k in range(k_steps):
        current = current @ trans_matrix
        rwse[:, k] = torch.diagonal(current)

    return rwse


class GPSLayer(nn.Module):
    """
    GPS (Graph Transformer with Positional Encoding and Structure) Layer.

    Combines local message passing with global attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize GPS layer.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Local message passing (GNN)
        self.local_gnn = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=2  # Edge features: [reward, depth_delta]
        )

        # Global attention (Transformer)
        self.global_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Combination weights
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, 2]
            batch: Batch assignment [num_nodes]

        Returns:
            Updated node features
        """
        # Local message passing
        local_out = self.local_gnn(x, edge_index, edge_attr)
        local_out = self.norm1(local_out)

        # Global attention (treat all nodes as a sequence)
        if batch is not None:
            # Handle batched graphs
            # For simplicity, we process each graph separately
            global_out = x.clone()
            for batch_id in torch.unique(batch):
                mask = batch == batch_id
                nodes_in_graph = x[mask].unsqueeze(0)  # [1, num_nodes_in_graph, hidden_dim]

                attn_out, _ = self.global_attention(
                    nodes_in_graph,
                    nodes_in_graph,
                    nodes_in_graph
                )
                global_out[mask] = attn_out.squeeze(0)
        else:
            # Single graph
            x_expanded = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            global_out, _ = self.global_attention(x_expanded, x_expanded, x_expanded)
            global_out = global_out.squeeze(0)

        global_out = self.norm2(global_out)

        # Combine local and global
        alpha = torch.sigmoid(self.alpha)
        combined = alpha * local_out + (1 - alpha) * global_out

        # Feed-forward network
        ffn_out = self.ffn(combined)
        output = self.norm3(combined + ffn_out)

        return output


class GPSPolicyNetwork(nn.Module):
    """
    Graph Transformer policy using GPS architecture.

    Architecture:
        1. Node feature projection: hidden_dim → policy_hidden_dim
        2. GPS layers (L=4): Local MPNN + Global Attention
        3. Readout: Current node embedding
        4. Action head: MLP → 4-dim action logits

    Input:
        - graph: PyG Data object with node/edge features
        - current_node_idx: Index of current position

    Output:
        - action_logits: [batch_size, 4] logits over actions
        - action_probs: [batch_size, 4] softmax probabilities
    """

    def __init__(
        self,
        input_dim: int = 4096,  # Qwen3 hidden dim
        hidden_dim: int = 512,
        num_layers: int = 2,  # Paper uses 2 GPS layers
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_actions: int = 5,
        use_rwse: bool = True,
        rwse_dim: int = 8
    ):
        """
        Initialize GPS policy network.

        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden dimension for GPS layers
            num_layers: Number of GPS layers (paper: 2)
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function
            num_actions: Number of actions (5: EXPAND, BRANCH, BACKTRACK, TERMINATE, SPAWN)
            use_rwse: Whether to use Random Walk Structural Encodings
            rwse_dim: Dimension of RWSE (number of random walk steps)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim

        # RWSE projection (if enabled)
        if use_rwse:
            self.rwse_projection = nn.Sequential(
                nn.Linear(rwse_dim, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout)
            )
            # Node feature projection (input + RWSE)
            self.node_projection = nn.Sequential(
                nn.Linear(input_dim + hidden_dim // 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.node_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout)
            )

        # GPS layers
        self.gps_layers = nn.ModuleList([
            GPSLayer(hidden_dim, num_heads, dropout, activation)
            for _ in range(num_layers)
        ])

        # Action head (from current node embedding)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )

        # Value head (for PPO critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        logger.info(f"Initialized GPS Policy Network: {num_layers} layers, {hidden_dim} hidden dim")

    def forward(
        self,
        graph: Data,
        return_value: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            graph: PyG Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge attributes [num_edges, 2]
                - current_node_idx: Current node index (int or Tensor)
                - batch (optional): Batch assignment
            return_value: Whether to return value estimate

        Returns:
            Tuple of (action_logits, action_probs, value)
        """
        # Compute RWSE if enabled
        if self.use_rwse:
            rwse = compute_random_walk_pe(
                graph.edge_index,
                graph.num_nodes,
                k_steps=self.rwse_dim
            )
            rwse_features = self.rwse_projection(rwse)

            # Concatenate with node features
            x_combined = torch.cat([graph.x, rwse_features], dim=-1)
            x = self.node_projection(x_combined)
        else:
            x = self.node_projection(graph.x)

        # Apply GPS layers
        for gps_layer in self.gps_layers:
            x = gps_layer(
                x,
                graph.edge_index,
                graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                graph.batch if hasattr(graph, 'batch') else None
            )

        # Get current node embedding
        if isinstance(graph.current_node_idx, torch.Tensor):
            if graph.current_node_idx.dim() == 0:
                current_node_idx = graph.current_node_idx.item()
            else:
                # Batched case
                current_node_idx = graph.current_node_idx
        else:
            current_node_idx = graph.current_node_idx

        current_node_embedding = x[current_node_idx]

        # Compute action logits
        action_logits = self.action_head(current_node_embedding)

        # Apply action masking (if provided)
        if hasattr(graph, 'action_mask'):
            action_mask = graph.action_mask
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))

        # Compute action probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        # Compute value estimate
        value = None
        if return_value:
            value = self.value_head(current_node_embedding)

        return action_logits, action_probs, value

    def get_action_distribution(
        self,
        graph: Data,
        temperature: float = 1.0
    ) -> torch.distributions.Categorical:
        """
        Get action distribution for sampling.

        Args:
            graph: PyG Data object
            temperature: Sampling temperature

        Returns:
            Categorical distribution over actions
        """
        action_logits, _, _ = self.forward(graph)

        # Apply temperature
        action_logits = action_logits / temperature

        # Create categorical distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)

        return action_dist

    def select_action(
        self,
        graph: Data,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action from policy.

        Args:
            graph: PyG Data object
            temperature: Sampling temperature
            deterministic: If True, select argmax action

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_dist = self.get_action_distribution(graph, temperature)

        if deterministic:
            action = action_dist.probs.argmax()
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)

        # Get value estimate
        _, _, value = self.forward(graph, return_value=True)

        return action.item(), log_prob, value

    def evaluate_actions(
        self,
        graph: Data,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            graph: PyG Data object
            actions: Actions taken [batch_size]

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, action_probs, values = self.forward(graph, return_value=True)

        # Create distribution
        action_dist = torch.distributions.Categorical(probs=action_probs)

        # Compute log probabilities
        log_probs = action_dist.log_prob(actions)

        # Compute entropy
        entropy = action_dist.entropy()

        return log_probs, values.squeeze(-1), entropy


def create_action_mask(tree_state, num_actions: int = 5) -> torch.Tensor:
    """
    Create action mask for current tree state.

    Args:
        tree_state: TreeState object
        num_actions: Number of actions

    Returns:
        Boolean mask [num_actions]
    """
    mask = torch.ones(num_actions, dtype=torch.bool)

    # Action indices: 0=EXPAND, 1=BRANCH, 2=BACKTRACK, 3=TERMINATE, 4=SPAWN

    # Can't backtrack from root
    if tree_state.current_node.is_root():
        mask[2] = False

    # If at max depth, can't expand or branch
    max_depth = 20  # From config
    if tree_state.current_node.depth >= max_depth:
        mask[0] = False
        mask[1] = False

    # SPAWN requires at least one non-root node to spawn from
    # Can't spawn if only root exists
    if len(tree_state.nodes) <= 1:
        mask[4] = False

    return mask
