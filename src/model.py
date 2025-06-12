from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPOutput:

    y: torch.Tensor


class MLP(nn.Module):

    def __init__(self, layers: list[nn.Module]):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(*layers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> MLPOutput:

        y = self.layers(x)

        return MLPOutput(y=y)
