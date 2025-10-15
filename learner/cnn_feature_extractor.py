from e2cnn import nn as e2nn
from torch import nn as nn

from learner.c6_conv_layer import C6ConvLayer


class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor using C6-equivariant convolutional layers.
    """

    def __init__(self, in_channels=4, hidden_channels=32):
        super(CNNFeatureExtractor, self).__init__()

        # First C6-equivariant convolutional layer
        self.c6_conv1 = C6ConvLayer(
            in_channels, hidden_channels, kernel_size=3, padding=1
        )

        # Second C6-equivariant convolutional layer
        self.c6_conv2 = C6ConvLayer(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

        # Pooling layer (equivariant)
        self.pool = e2nn.PointwiseMaxPool(
            self.c6_conv2.conv.out_type, kernel_size=2, stride=2
        )

    def forward(self, x):
        """
        Forward pass for the CNN feature extractor.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
        Returns:
            GeometricTensor: Extracted feature maps after convolutions and pooling.
        """
        # Initialize GeometricTensor with input field type
        x = e2nn.GeometricTensor(x, self.c6_conv1.conv.in_type)

        # Apply first C6 convolutional layer
        x = self.c6_conv1(x)

        # Apply second C6 convolutional layer
        x = self.c6_conv2(x)

        # Apply pooling
        x = self.pool(x)

        return x  # GeometricTensor with updated spatial dimensions
