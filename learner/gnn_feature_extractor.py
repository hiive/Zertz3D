from torch import nn as nn
from torch_geometric.nn import GCNConv

# PyTorch Geometric for GNNs
class GNNFeatureExtractor(nn.Module):
    """
    GNN Feature Extractor using Graph Convolutional Networks (GCNs).
    """

    def __init__(self, in_channels=32, hidden_channels=64, num_layers=2):
        super(GNNFeatureExtractor, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """
        Forward pass for the GNN feature extractor.
        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.LongTensor): Edge indices in COO format [2, num_edges].
        Returns:
            torch.Tensor: Updated node features after GNN layers.
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
        return x  # Shape: [num_nodes, hidden_channels]
