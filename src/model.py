from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPOutput:

    y: torch.Tensor
    loss: torch.Tensor


class MLP(nn.Module):

    def __init__(self, layers: list[nn.Module], loss_function: nn.Module):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(*layers)
        self.loss_function = loss_function

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> MLPOutput:

        y = self.layers(x)

        # `loss_function(z, x)` computes tensor of shape [z, x] with element-wise loss
        # `.sum(-1)` sums loss over predicted features (total loss per sample)
        # `.mean()` averages the per sample loss over batch
        loss = self.loss_function(y, x).sum(-1).mean()

        return MLPOutput(y=y, loss=loss)
