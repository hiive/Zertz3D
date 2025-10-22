import torch
from torch import nn as nn
from torch.nn import functional as F

from learner.cnn_feature_extractor import CNNFeatureExtractor
from learner.gnn_feature_extractor import GNNFeatureExtractor


class HybridZertzAI(nn.Module):
    """
    Hybrid Neural Network combining C6-equivariant CNN and GNN for ZÈRTZ AI.
    Outputs policy (move probabilities for PUT-REM and CAP actions) and value (position evaluation).
    """

    def __init__(
        self,
        in_channels=4,
        cnn_hidden=32,
        gnn_hidden=64,
        num_moves=48,
        height=8,
        width=8,
        num_captured_features=6,
    ):
        """
        Args:
            in_channels (int): Number of input channels (4: ring, w, g, b).
            cnn_hidden (int): Number of hidden channels in CNN layers.
            gnn_hidden (int): Number of hidden channels in GNN layers.
            num_moves (int): Total number of possible moves (number of cells on the board).
            height (int): Number of rows in the board grid (fixed at 8 for 48-ring board).
            width (int): Number of columns in the board grid (fixed at 8 for 48-ring board).
            num_captured_features (int): Number of captured marble count features (6: p1_w, p1_g, p1_b, p2_w, p2_g, p2_b).
        """
        super(HybridZertzAI, self).__init__()

        self.height = height
        self.width = width
        self.num_moves = num_moves
        self.num_captured_features = num_captured_features

        # CNN Component
        self.cnn = CNNFeatureExtractor(
            in_channels=in_channels, hidden_channels=cnn_hidden
        )

        # GNN Component
        self.gnn = GNNFeatureExtractor(
            in_channels=cnn_hidden, hidden_channels=gnn_hidden, num_layers=2
        )

        # Captured Counts Embedding
        self.captured_embed = nn.Linear(num_captured_features, gnn_hidden)

        # Policy Heads
        self.fc_policy_put = nn.Linear(
            gnn_hidden, num_moves
        )  # For Placement and Removal
        self.fc_policy_cap = nn.Linear(gnn_hidden, num_moves)  # For Capture actions

        # Value Head
        self.fc_value = nn.Linear(gnn_hidden, 1)

    def forward(self, x, edge_index, captured_counts, action_type="put"):
        """
        Forward pass for the hybrid network.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
            edge_index (torch.LongTensor): Edge connections for the graph.
            captured_counts (torch.Tensor): Captured marble counts of shape [batch_size, num_captured_features].
            action_type (str): Type of action to predict ('put' or 'cap').
        Returns:
            policy (torch.Tensor): Move probabilities of shape [batch_size, num_moves].
            value (torch.Tensor): Position evaluations of shape [batch_size, 1].
        """
        batch_size = x.size(0)

        # CNN Feature Extraction
        cnn_features = self.cnn(
            x
        )  # GeometricTensor with shape [batch_size, cnn_hidden, H', W']

        # Prepare node features for GNN
        # Flatten the spatial dimensions to create node embeddings
        # Assuming pooling reduces height and width by factor of 2
        H_pooled = self.height // 2
        W_pooled = self.width // 2
        num_nodes = H_pooled * W_pooled  # Number of nodes after pooling

        # Reshape CNN features to [batch_size, num_nodes, cnn_hidden]
        # cnn_features.tensor shape: [batch_size, cnn_hidden, H', W']
        cnn_features = cnn_features.tensor.view(
            batch_size, -1, cnn_features.tensor.size(1)
        )

        # Pass through GNN and incorporate captured counts
        policies = []
        values = []

        for i in range(batch_size):
            # Get node features for the i-th sample
            node_features = cnn_features[i]  # Shape: [num_nodes, cnn_hidden]

            # Pass through GNN
            gnn_features = self.gnn(
                node_features, edge_index
            )  # Shape: [num_nodes, gnn_hidden]

            # Global pooling (mean pooling)
            pooled_features = gnn_features.mean(dim=0)  # Shape: [gnn_hidden]

            # Incorporate captured counts
            captured = captured_counts[i]  # Shape: [num_captured_features]
            captured_embedding = self.captured_embed(captured)  # Shape: [gnn_hidden]

            # Combine pooled features with captured counts
            combined_features = (
                pooled_features + captured_embedding
            )  # Shape: [gnn_hidden]

            # Policy head based on action type
            if action_type == "put":
                policy = self.fc_policy_put(combined_features)  # Shape: [num_moves]
            elif action_type == "cap":
                policy = self.fc_policy_cap(combined_features)  # Shape: [num_moves]
            else:
                raise ValueError("Invalid action_type. Choose 'put' or 'cap'.")

            policy = F.softmax(policy, dim=0)  # Convert to probabilities

            # Value head
            value = self.fc_value(combined_features)  # Shape: [1]
            value = torch.tanh(value)  # Scale between -1 and 1

            policies.append(policy)
            values.append(value)

        # Stack policies and values
        policies = torch.stack(policies)  # Shape: [batch_size, num_moves]
        values = torch.stack(values)  # Shape: [batch_size, 1]

        return policies, values
