from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import yaml

import torch.nn as nn
import torch.optim as optim

torch_layers = {"linear": nn.Linear}
torch_activation = {"relu": nn.ReLU}
torch_loss = {"mse": nn.MSELoss}
torch_optim = {"adam": optim.Adam}


@dataclass(init=False)
class DataConfig:

    trainig_data: Path
    test_data: Path
    imputation_mask: Path

    # Custom field to store any extra keyword arguments not in the declared fields
    custom: Any = field(init=False)

    def __init__(self, **kwargs: Any):
        # Get names of all declared fields except 'custom'
        declared_fields = set(self.__class__.__dataclass_fields__) - {"custom"}

        # Initialize an empty SimpleNamespace to hold extra parameters
        self.custom = SimpleNamespace()

        for key, value in kwargs.items():
            if key in declared_fields:
                setattr(self, key, value)
            else:
                setattr(self.custom, key, value)


@dataclass
class ModelLayer:

    type: str
    in_dim: int
    out_dim: int
    activation: str

    def to_torch(self):
        torch_layer = []
        torch_layer.append(torch_layers[self.type](self.in_dim, self.out_dim))
        torch_layer.append(torch_activation[self.activation]())

        return torch_layer


@dataclass
class ModelConfig:

    name: str
    type: str
    n_layers: int
    raw_layers: list
    loss: nn.Module
    layers: list = field(init=False)

    def __post_init__(self):
        super().__setattr__("layers", self.process_layers(self.type, self.raw_layers))
        self.loss = torch_loss[self.loss]

    def process_layers(self, type, raw_layers):

        if type == "mlp":
            return self.assemble_mlp(raw_layers)

    def assemble_mlp(self, raw_layers):

        assembled = []
        for layer in raw_layers.values():
            assembled.append(ModelLayer(**layer))

        return assembled

    def to_torch(self):

        torch_layers = []
        for layer in self.layers:
            torch_layers.extend(layer.to_torch())

        return torch_layers


@dataclass
class TrainingConfig:

    kfolds: int
    batch_size: int
    epochs: int
    optimizer: optim.Optimizer

    def __post_init__(self):
        self.optimizer = torch_optim[self.optimizer]


def load_config_yml(path_to_file: str) -> dict:

    with open(path_to_file, "r") as yml:
        config = yaml.safe_load(yml)

    return config


def setup_classes(path_to_file: str):

    config = load_config_yml(path_to_file)

    if "data" in config:
        data_config = DataConfig(**config["data"])

    if "model" in config:
        model_config = ModelConfig(**config["model"])

    if "training" in config:
        training_config = TrainingConfig(**config["training"])

    return data_config, model_config, training_config
