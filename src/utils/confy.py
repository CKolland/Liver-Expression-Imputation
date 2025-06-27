from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Union

import torch.nn as nn
import torch.optim as optim

import fit.constants as C

# --- Dataset configuration classes ---


@dataclass
class DataCSV:
    """Configuration for a CSV dataset.

    This class represents the configuration required to load a ".csv" dataset. It includes
    the path to the file, the dataset type (which defaults to "csv"), and an optional header
    flag that determines whether the file includes a header row.

    :param str path: Path to the ".csv" file.
    :param str type: Type identifier for the dataset, defaults to "csv".
    :param Optional[int] header: Indicates if the CSV has a header row.
                                 Use 0 if a header is present, or None otherwise.
    """

    path: str
    type: str = "csv"
    header: Optional[int] = None

    def __post_init__(self) -> None:
        """Post-initialization processing.

        Converts the 'header' attribute from string "none" to None to standardize
        the representation of missing headers.
        """
        # Normalize string input for 'header' to appropriate None value
        if self.header == "none":
            self.header = None


@dataclass
class DataH5AD:
    """Configuration for an ".h5ad" dataset, typically used in single-cell analysis.

    :param str path: Path to the ".h5ad" file.
    :param str type: Type identifier, defaults to "h5ad".
    :param Optional[str] layer: Specific layer within the h5ad file (e.g., "counts").
    """

    path: str
    type: str = "h5ad"
    layer: Optional[str] = None


@dataclass
class DataCollection:
    """Handles a collection of dataset configurations, auto-instantiating them as
    either `DataCSV` or `DataH5AD` instances based on their type.

    :param dict config: Mapping of dataset names to configuration dictionaries.

    :raises ValueError: If an unsupported dataset type is encountered.
    """

    config: dict[str, dict] = field()

    def __post_init__(self):
        # Convert each dataset dict into its corresponding dataclass
        for key, dataset in self.config.items():
            if dataset["type"] == "csv":
                setattr(self, key, DataCSV(**dataset))
            elif dataset["type"] == "h5ad":
                setattr(self, key, DataH5AD(**dataset))
            else:
                raise ValueError(f"Unsupported dataset type: {dataset['type']}")

    def __repr__(self) -> str:
        """
        String representation excluding the 'config' dictionary.

        :return str: Readable format showing all dataset attributes.
        """
        attrs = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if k != "config"
        )
        return f"{self.__class__.__name__}({attrs})"


# --- Layer and Model configuration ---


@dataclass
class LayerConfig:
    """Represents configuration for a single model layer.

    :param str type: Layer type (e.g., "linear").
    :param int in_dim: Number of input features.
    :param int out_dim: Number of output features.
    :param Optional[str] activation: Activation function name (e.g., "relu").
    :param float dropout: Optional dropout probability between 0.0 and 1.0.
    """

    type: str
    in_dim: int
    out_dim: int
    activation: Optional[str] = None
    dropout: float = 0.0  # Default is no dropout

    def to_torch(self):
        """Converts this layer configuration to a list of PyTorch modules.

        :return List[nn.Module]: Sequential layer components.
        """
        layer = []
        layer.append(get_layer(self.type)(self.in_dim, self.out_dim))

        if self.dropout > 0.0:
            layer.append(get_layer("dropout")(p=self.dropout))

        if self.activation is not None:
            layer.append(get_activation(self.activation)())

        return layer


@dataclass
class ModelConfig:
    """Represents the full model configuration including metadata and layers.

    :param str name: Name of the model.
    :param str type: Type identifier (e.g., "mlp").
    :param int n_layers: Number of layers in the model.
    :param List[LayerConfig] layers: List of layer configurations.
    """

    name: str
    type: str
    n_layers: int
    layers: list[LayerConfig] = field(default_factory=list)

    def to_torch(self):
        """Constructs a list of PyTorch modules from the defined layer configurations.

        :return List[nn.Module]: List of PyTorch layers ready for nn.Sequential.
        """
        torch_layers = []
        for layer in self.layers:
            torch_layers.extend(layer.to_torch())

        return torch_layers


# --- Training configuration ---


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    patience: int = 20
    delta: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    @staticmethod
    def get_optimizer(name: str) -> optim.Optimizer:
        """_summary_

        :param name: _description_
        :type name: str
        :return: _description_
        :rtype: optim.Optimizer
        """
        optims = C.TORCH_OPTIM

        if name not in optims:
            raise ValueError(f"Unsupported optimizer: {name}")

        return optims[name]

    def create_optimizer(self, model_params) -> optim.Optimizer:
        """Create optimizer instance with this configuration."""
        if self.optimizer.lower() in ["adam", "adamw"]:
            return self._get_optimizer(self.optimizer)(
                model_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )


@dataclass
class TrainingConfig:
    """Main training configuration."""

    seed: int
    loss: nn.Module
    kfolds: int = 5
    batch_size: int = 128
    epochs: int = 200
    num_workers: int = 4
    early_stopping: EarlyStoppingConfig = None
    optimization: OptimizationConfig = None


# --- Factory functions ---


def get_activation(name: str) -> nn.Module:
    """Maps an activation name to its corresponding PyTorch class.

    :param str name: Name of the activation function (e.g., "relu").

    :return nn.Module: Corresponding PyTorch activation class.

    :raises ValueError: If the activation name is not found.
    """
    # Available functions stored in constants for easy access
    available = C.TORCH_ACTIVATIONS

    if name not in available:
        raise ValueError(f"Unsupported activation type: {name}")

    return available[name]


def get_layer(name: str) -> nn.Module:
    """Maps a layer type name to its corresponding PyTorch layer constructor.

    :param str name: Name of the layer (e.g., "linear", "dropout").

    :return nn.Module: Corresponding PyTorch layer class.

    :raises ValueError: If the layer name is not found.
    """
    # Available layers stored in constants for easy access
    available = C.TORCH_LAYERS

    if name not in available:
        raise ValueError(f"Unsupported layer type: {name}")

    return available[name]


def get_loss(name: str) -> nn.Module:
    """_summary_

    :param name: _description_
    :type name: str
    :raises ValueError: _description_
    :return: _description_
    :rtype: nn.Module
    """
    available = C.TORCH_LOSS

    if name not in available:
        raise ValueError(f"Unsupported loss function: {name}")

    return available[name]


def setup_dataset(dataset: dict) -> Union[DataCSV, DataH5AD, DataCollection]:
    """Factory function to instantiate a dataset object or a collection of datasets.

    - If the input is a single dataset dictionary with "path" and "type",
      it returns a `DataCSV` or `DataH5AD` instance.
    - Otherwise, it assumes a collection of datasets and returns a DatasetCollection.

    :param dict dataset: Dictionary containing dataset configuration(s).

    :return Union[DataCSV, DataH5AD, DatasetCollection]: An appropriate dataset instance.

    :raises ValueError: If the dataset type is unsupported.
    """
    # Handle single dataset configuration
    if isinstance(dataset, dict) and {"path", "type"}.issubset(dataset.keys()):
        if dataset["type"] == "csv":
            return DataCSV(**dataset)
        elif dataset["type"] == "h5ad":
            return DataH5AD(**dataset)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset['type']}")
    # Handle dataset collection
    else:
        return DataCollection(dataset)


def setup_model(model: dict) -> ModelConfig:
    """Creates a ModelConfig object from a dictionary by parsing its layer definitions.

    :param dict model: Dictionary containing the model structure and layers.

    :return ModelConfig: Parsed and validated model configuration object.
    """
    model_copy = deepcopy(model)

    layers = []
    for _, value in model_copy["layers"].items():
        layer_config = LayerConfig(**value)
        layers.append(layer_config)

    model_copy["layers"] = layers

    return ModelConfig(**model_copy)


def setup_training(training: dict) -> TrainingConfig:
    """_summary_

    :param training: _description_
    :type training: dict
    :return: _description_
    :rtype: TrainingConfig
    """
    training_copy = deepcopy(training)

    # Create early stopping config
    if "early_stopping" in training_copy.keys():
        early_stopping = EarlyStoppingConfig(**training_copy["early_stopping"])
        training_copy["early_stopping"] = early_stopping

    if "optimization" in training_copy.keys():
        optimization = OptimizationConfig(**training_copy["optimization"])
        training_copy["optimization"] = optimization

    loss = get_loss(training_copy["loss"])
    training_copy["loss"] = loss

    return TrainingConfig(**training_copy)
