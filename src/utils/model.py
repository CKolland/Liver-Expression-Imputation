import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple multi-layer perceptron (MLP) model using a sequential stack of layers.

    :param list[nn.Module] layers: List of PyTorch layers to include in the MLP.
    :param str device: Device to run the model on (e.g., "cpu" or "cuda").
    """

    def __init__(self, layers: list[nn.Module], device: str):
        super(MLP, self).__init__()

        # Compose the layers into a single sequential model
        self.layers = nn.Sequential(*layers)

        self.device = device  # Store the device information
        self.to(self.device)  # Move the model to the specified device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        :param torch.Tensor x: Input tensor.

        :return torch.Tensor: Output tensor after passing through the layers.
        """
        return self.layers(x)

    def reset_weights(self):
        """
        Reset model weights to avoid weight leakage.
        """
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
