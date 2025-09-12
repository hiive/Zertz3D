import torch.nn as nn

# e2cnn for Group Equivariant CNNs
from e2cnn import gspaces
from e2cnn import nn as e2nn




class C6ConvLayer(nn.Module):
    """
    C6-equivariant Convolutional Layer using e2cnn's R2Conv.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(C6ConvLayer, self).__init__()

        # Define the C6 symmetry group (rotations by multiples of 60 degrees)
        self.r2_act = gspaces.Rot2dOnR2(N=6)

        # Define input and output field types
        # Each channel is associated with a regular representation of C6
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.regular_repr] * in_channels)
        out_type = e2nn.FieldType(self.r2_act, [self.r2_act.regular_repr] * out_channels)

        # Define the equivariant convolution
        self.conv = e2nn.R2Conv(
            in_type,
            out_type,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

        # Define equivariant batch normalization
        self.bn = e2nn.InnerBatchNorm(out_type)

        # Define equivariant activation function
        self.relu = e2nn.ReLU(out_type, inplace=True)

    def forward(self, x):
        """
        Forward pass for the C6-equivariant convolutional layer.
        Args:
            x (GeometricTensor): Input tensor with associated field type.
        Returns:
            GeometricTensor: Output tensor after convolution, batch norm, and activation.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


