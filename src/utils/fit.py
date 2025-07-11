import copy
import gc
from logging import Logger
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from tqdm import tqdm

from utils.io import ImputationDataset, TrainingMetrics


class EarlyStopping:
    """Utility for early stopping to prevent overfitting during training.

    Monitors the validation loss and halts training if the loss does not improve
    after a specified number of epochs. Saves the best model state for later use.
    """

    def __init__(self, patience: int = 20, delta: float = 0.0):
        """Initialize EarlyStopping.

        :param int patience: Number of epochs to wait without improvement before stopping.
        :type patience: int
        :param delta: Minimum change in validation loss to be considered improvement.
        :type delta: float
        """
        self.patience: int = patience
        self.delta: float = delta
        self.counter: int = 0
        self.best_val_loss: float | None = None
        self.best_model_state: dict[str, Any] | None = None
        self.early_stop: bool = False

    def __call__(self, val_loss: float, model: nn.Module):
        """Check for early stopping condition.

        :param val_loss: Current epoch's validation loss.
        :type val_loss: float
        :param model: Model to monitor.
        :type model: nn.Module

        :raises TypeError: If inputs are of incorrect types.
        """
        if not isinstance(val_loss, (float, int)):
            raise TypeError("`val_loss` must be a float or int.")
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be an instance of nn.Module.")

        score = -val_loss  # Lower loss = better score

        if self.best_val_loss is None:  # Initial validation loss
            self.best_val_loss = score
            self.best_model_state = model.state_dict().copy()
        elif score < self.best_val_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

    def load_best_model(self, model: nn.Module):
        """Restore the model to the best saved state.

        :param nn.Module model: Model to restore state into.

        :raises RuntimeError: If no best model state has been saved yet.
        :raises TypeError: If model is not a PyTorch module.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of nn.Module.")
        if self.best_model_state is None:
            raise RuntimeError("No saved model state. Cannot load best model.")

        model.load_state_dict(self.best_model_state)

    def reset(self):
        """Reset internal state for reuse in another training run."""
        self.counter = 0
        self.best_val_loss = None
        self.best_model_state = None
        self.early_stop = False


class TrainingPipeline:
    """Pipeline to train a model using K-fold cross-validation with early stopping.

    This pipeline handles the training of a PyTorch model using K-fold
    cross-validation and early stopping to prevent overfitting. It supports
    customizable optimizers, loss functions, and data loading parameters.
    """

    def __init__(
        self,
        dataset: Dataset | ImputationDataset,
        model: nn.Module,
        device: torch.device,
        logger: Logger,
        kfolds: int = 5,
        epochs: int = 200,
        patience: int = 20,
        delta: float = 0.0,
        batch_size: int = 128,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: optim.Optimizer = optim.Adam,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        use_scheduler: bool = False,
        seed: int | None = None,
        num_workers: int = 4,
    ):
        """Initialize the training pipeline with cross-validation and early stopping.

        :param dataset: Dataset used for training, must be compatible with torch.utils.data.Dataset.
        :type dataset: Dataset | ImputationDataset
        :param model: PyTorch model to be trained.
        :type model: nn.Module
        :param device: Device on which to perform training (e.g., CPU or CUDA).
        :type device: torch.device
        :param logger: Logger instance to record training metrics and progress.
        :type logger: Logger
        :param kfolds: Number of K-folds for cross-validation. Must be >= 2.
        :type kfolds: int
        :param epochs: Maximum number of training epochs per fold.
        :type epochs: int
        :param patience: Number of epochs with no improvement after which training will be stopped early.
        :type patience: int
        :param delta: Minimum change in validation loss to qualify as an improvement.
        :type delta: float
        :param batch_size: Number of samples per training batch.
        :type batch_size: int
        :param criterion: Loss function to be used during training.
        :type criterion: nn.Module
        :param optimizer: Optimizer class (not instance) to use for training.
        :type optimizer: type[optim.Optimizer]
        :param learning_rate: Learning rate for the optimizer.
        :type learning_rate: float
        :param weight_decay: Weight decay (L2 regularization) coefficient.
        :type weight_decay: float
        :param seed: Optional random seed for reproducibility.
        :type seed: int | None
        :param num_workers: Number of subprocesses to use for data loading.
        :type num_workers: int
        """
        self.dataset: Dataset | ImputationDataset = dataset
        self.model: nn.Module = model
        self.device: torch.device = device
        self.logger: Logger = logger
        self.kfolds: int = kfolds
        self.epochs: int = epochs
        self.patience: int = patience
        self.delta: float = delta
        self.early_stopping: EarlyStopping | None = None
        self.batch_size: int = batch_size
        self.criterion: nn.Module = criterion
        self.optimizer: optim.Optimizer = optimizer
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.use_scheduler: bool = use_scheduler
        self.seed = seed
        self.num_workers: int = num_workers

    def _setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Initialize optimizer with model parameters.

        :param model: Model to optimize.
        :type model: nn.Module

        :return: Configured optimizer instance.
        :rtype: optim.Optimizer
        """
        if self.optimizer == optim.Adam:
            optimizer = self.optimizer(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        if self.optimizer == optim.AdamW:
            optimizer = self.optimizer(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        return optimizer

    def _clear_gpu_memory(self):
        """Clear GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def _train_epoch(
        self,
        fold: int,
        epoch: int,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
    ) -> tuple[float, float, float]:
        """Train the model for a single epoch and return training statistics.

        Executes a single training epoch over the provided `train_loader`. For each batch,
        it performs the forward pass, computes the loss, backpropagates the gradients,
        updates model weights, and tracks gradient statistics. The method computes:

        - Average training loss across all batches
        - Average L2 norm of gradients across all batches
        - Maximum absolute gradient value observed in the epoch

        This function is typically used within a K-fold cross-validation loop.

        :param fold: Index of the current fold in K-fold cross-validation (0-based).
        :type fold: int
        :param int epoch: Index of the current epoch (0-based).
        :type epoch: int
        :param DataLoader train_loader: PyTorch DataLoader yielding training batches.
        :type train_loader: DataLoader
        :param optim.Optimizer optimizer: Optimizer instance used to update model parameters.
        :type optimizer: optim.Optimizer

        :return: A tuple containing:
            (1) Average loss across all training batches
            (2) Average L2 norm of gradients
            (3) Maximum absolute gradient value seen in any parameter
        :rtype: tuple[float, float, float]
        """
        self.model.train()  # Set model to training mode
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_max_grad = 0.0
        num_batches = len(train_loader)

        # Create a progress bar to monitor training
        pbar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc=f"Fold {fold + 1} - Train Epoch {epoch + 1}",
        )
        for batch_idx, (x, y) in pbar:
            # Move input (x) and target (y) to the specified device
            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()  # Clear previous gradients

            # Forward pass to create prediction (y_hat)
            y_hat = self.model(x)

            loss = self.criterion(y_hat, y).sum(-1).mean()  # Compute average batch loss
            loss.backward()  # Backpropagation

            # Compute L2 norm and maximum of gradients for monitoring
            batch_max_grad = 0.0
            total_norm_squared = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm_squared = param.grad.norm(2) ** 2
                    total_norm_squared += param_norm_squared

                    param_max_grad = param.grad.abs().max().item()
                    batch_max_grad = max(batch_max_grad, param_max_grad)

            optimizer.step()  # Update model parameters

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_grad_norm += (total_norm_squared**0.5).item()
            epoch_max_grad = max(epoch_max_grad, batch_max_grad)

            # Update progress bar with current batch loss
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # Return average loss, average gradient norm, and maximum gradient seen
        return (
            epoch_loss / len(train_loader),
            epoch_grad_norm / len(train_loader),
            epoch_max_grad,
        )

    def _validate_epoch(
        self,
        fold: int,
        epoch: int,
        val_loader: DataLoader,
    ) -> float:
        """Validate the model for a single epoch and return the average loss.

        This method evaluates the model performance on the validation set provided
        by `val_loader`. It disables gradient computations for efficiency and sets
        the model to evaluation mode to ensure correct behavior (e.g., for dropout
        and batch normalization layers). The average validation loss is calculated
        across all batches in the epoch.

        :param fold: Current fold number in cross-validation (0-indexed).
        :type fold: int
        :param int epoch: Current epoch number (0-indexed).
        :type epoch: int
        :param DataLoader val_loader: PyTorch DataLoader providing validation batches.
        :type val_loader: DataLoader

        :return: Average validation loss over all batches
        :rtype: float
        """
        self.model.eval()  # Set model to evaluation mode
        epoch_loss = 0.0

        # Disable gradient calculations for validation to save memory and speed up computation
        with torch.no_grad():
            # Create a progress bar to monitor validation progress
            pbar = tqdm(val_loader, desc=f"Fold {fold + 1} - Val Epoch {epoch + 1}")
            for x, y in pbar:
                # Move input and target tensors to the specified device
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model(x)  # Forward pass
                loss = self.criterion(y_hat, y).sum(-1).mean()  # Average batch loss

                # Accumulate batch loss
                epoch_loss += loss.item()

                # Update progress bar with current batch loss
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # Return average validation loss across all batches
        return epoch_loss / len(val_loader)

    def fit_kfold(
        self,
        metrics: TrainingMetrics,
    ) -> tuple[nn.Module, TrainingMetrics]:
        """
        Train the model using K-fold cross-validation with metric tracking and early stopping.

        Performs K-fold cross-validation using the configured dataset, model, optimizer,
        and training parameters. For each fold, a new model instance is trained with early
        stopping, and validation metrics are recorded. The best-performing model based on
        validation loss is selected at the end.

        Training progress and metrics for each fold and epoch are logged using the provided
        logger. The final result includes the best model (by validation loss) and the
        complete metrics history across all folds.

        :param metrics: Container to accumulate training and validation metrics across folds.
        :type metrics: TrainingMetrics

        :return: A tuple containing:
            (1) The best-performing model after cross-validation.
            (2) The populated metrics object with full training history.
        :rtype: tuple[nn.Module, TrainingMetrics]
        """
        self.logger.info(f"Starting {self.kfolds}-fold cross-validation...")

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.logger.debug(f"Train model based on seed: {self.seed}")

        # Initialize KFold splitter with deterministic shuffling if seed is provided
        kfold = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            self.logger.info(f"----")
            self.logger.info(f"Fold {fold + 1}/{self.kfolds}")
            self.logger.info(
                f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}"
            )

            # Clear GPU memory before starting each fold
            self._clear_gpu_memory()

            # Create data loaders with subset samplers for training and validation
            train_loader = DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                sampler=SubsetRandomSampler(train_idx),
                num_workers=self.num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            # Subset (no shuffling needed)
            val_dataset = Subset(self.dataset, val_idx)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            # Reset weights of model in fresh fold
            self.model.reset_weights()

            # Create optimizer for current model parameters
            optimizer = self._setup_optimizer(self.model)

            if self.use_scheduler:
                scheduler = ReduceLROnPlateau(optimizer, "min")
                self.logger.debug(f"Use scheduler on fold {fold + 1}.")
            else:
                scheduler = None

            # Initialize early stopping for this fold
            early_stopping = EarlyStopping(patience=self.patience, delta=self.delta)

            # Epoch-wise training and validation loop
            for epoch in range(self.epochs):
                train_loss, grad_norm, max_grad = self._train_epoch(
                    fold,
                    epoch,
                    train_loader,
                    optimizer,
                )
                val_loss = self._validate_epoch(
                    fold,
                    epoch,
                    val_loader,
                )

                # Make learning rate step according to validation loss
                if scheduler is not None:
                    scheduler.step(val_loss)

                # Record metrics
                metrics.add_fold_epoch(
                    fold + 1,
                    epoch + 1,
                    train_loss,
                    grad_norm,
                    max_grad,
                    val_loss,
                )

                self.logger.info(
                    f"Fold {fold + 1}, Epoch {epoch + 1}: "
                    f"Train Loss: {train_loss:.6f}, Grad Norm: {grad_norm:.6f}, "
                    f"Max Grad: {max_grad:.6f} Val Loss: {val_loss:.6f}"
                )

                # Trigger early stopping if no improvement
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    self.logger.info(f"Early stopping triggered at Epoch {epoch + 1}")
                    break

                # Clear GPU memory after each epoch
                self._clear_gpu_memory()

            # Determine best model for the fold based on validation loss
            metrics.update_best_model(
                fold + 1,
                early_stopping.best_val_loss,
                early_stopping.best_model_state,
            )

            self.logger.info(
                f"Fold {fold + 1} completed. Best val loss: {abs(early_stopping.best_val_loss):.6f}"
            )

        # Load the best-performing model weights into a new model instance
        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(metrics.best_model_state)

        self.logger.info("----")
        self.logger.info(f"✅ Cross-validation completed!")
        self.logger.info(
            f"Best model from fold {metrics.best_fold + 1} with val loss: {abs(metrics.best_val_loss):.6f}"
        )

        return best_model, metrics

    def save_training_results(
        self,
        model: nn.Module,
        metrics: TrainingMetrics,
        save_path: Path,
    ):
        """
        Save the trained model and associated training metrics to disk.

        Stores the model's state dictionary, detailed per-epoch metrics, and a
        fold-level summary in the specified directory. Metrics are saved as
        Apache Feather files for efficient read/write operations. If saving fails,
        an error is logged without raising an exception.

        :param model: Trained PyTorch model to be saved.
        :type model: nn.Module
        :param metrics: Object containing training and validation metrics.
        :type metrics: TrainingMetrics
        :param save_path: Target directory for saving model and metrics.
        :type save_path: Path
        """
        try:
            # Save the best model weights
            torch.save(model.state_dict(), save_path / "best_model.pth")

            # Convert and save detailed metrics as a feather file
            detailed_metrics = metrics.to_data_frame()
            detailed_metrics.to_feather(save_path / "training_metrics.feather")

            # Convert and save summary metrics for each fold
            summary_metrics = metrics.get_fold_summary()
            summary_metrics.to_feather(save_path / "fold_summary.feather")

            self.logger.info(f"Training results saved to: '{save_path}'")

        except Exception as e:
            # Log error if saving fails, but do not raise
            self.logger.error(f"Failed to save training results: {e}")


class TestingPipeline:
    """Testing pipeline for neural network model evaluation and result storage.

    This class provides functionality to run model testing on a dataset,
    collect predictions, and organize results into structured formats including
    AnnData objects for downstream analysis.

    :param test_loader: DataLoader containing the test dataset
    :type test_loader: DataLoader
    :param model: Neural network model to be tested
    :type model: nn.Module
    :param device: Device to run the model on (CPU or GPU)
    :type device: torch.device
    :param logger: Optional logger for tracking testing progress and results
    :type logger: Logger | None
    """

    def __init__(
        self,
        test_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        cell_types: list[str] | None = None,
        logger: Logger | None = None,
    ):
        """Initialize the testing pipeline with model and data configuration."""
        self.test_loader: DataLoader = test_loader
        self.cell_types: list[str] | None = cell_types  # Optional cell type annotations
        self.model: nn.Module = model
        self.device: torch.device = device
        self.logger: Logger | None = logger

        # Storage containers for test results
        self.inputs = []
        self.targets = []
        self.predictions = []

    def test(self) -> dict[str, Any]:
        """Run the model testing pipeline on the test dataset.

        Performs inference on the entire test dataset, collecting inputs,
        targets, and predictions. The model is set to evaluation mode and
        gradients are disabled for efficient inference.

        :return: Dictionary containing test results with keys 'input', 'target',
                'prediction', and 'cell_type'
        :rtype: dict[str, Any]
        """
        self.model.eval()  # Set model to evaluation mode (disables dropout, batch norm training)

        # Reset storage containers for fresh test run
        self.inputs.clear()
        self.targets.clear()
        self.predictions.clear()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Test"):
                # Handle different batch formats - some datasets return tuples, others single tensors
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1] if len(batch) > 1 else None
                else:
                    x, y = batch, None

                # Move tensors to the specified device (CPU/GPU)
                x = x.to(self.device)
                if y is not None:
                    y = y.to(self.device)

                # Forward pass through the model
                y_hat = self.model(x)

                # Convert tensors to CPU and then to lists for storage
                batch_inputs = x.cpu().tolist()
                batch_targets = y.cpu().tolist()
                batch_predictions = y_hat.cpu().tolist()

                # Accumulate results across batches
                self.inputs.extend(batch_inputs)
                self.targets.extend(batch_targets)
                self.predictions.extend(batch_predictions)

        # Compile results into a structured dictionary
        if self.cell_types is not None:
            results = {
                "input": self.inputs,
                "target": self.targets,
                "prediction": self.predictions,
                "cell_type": self.cell_types,
            }
        else:
            results = {
                "input": self.inputs,
                "target": self.targets,
                "prediction": self.predictions,
            }

        return results

    def create_anndata(self, results: dict[str, Any]) -> ad.AnnData:
        """Create an AnnData object from the prediction results.

        Converts the testing results into an AnnData object, which is a standard
        format for storing annotated data matrices in computational biology.
        The input data serves as the main data matrix (X), while targets and
        predictions are stored in the multi-dimensional observations (obsm).

        :param results: Dictionary containing test results with keys 'input',
                       'target', 'prediction', and 'cell_type'
        :type results: dict[str, Any]

        :return: AnnData object containing all results with proper annotations
        :rtype: ad.AnnData

        :raises AttributeError: If logger is None and logging is attempted
        """
        if self.logger is not None:
            self.logger.info("Creating AnnData object from results.")

        # Use inputs as the main data matrix (X)
        X = csr_matrix(results["input"])

        if self.cell_types is not None:
            # Create observations (obs) dataframe with cell type annotations
            obs = pd.DataFrame({"cell_type": results["cell_type"]})
            obs.index = [f"sample_{i}" for i in range(len(obs))]

            # Create AnnData object
            adata = ad.AnnData(X=X, obs=obs)
        else:
            adata = ad.AnnData(X=X)

        # Store outputs and predictions in obsm (multi-dimensional observations)
        adata.obsm["targets"] = csr_matrix(results["target"])
        adata.obsm["predictions"] = csr_matrix(results["prediction"])

        # Add metadata to uns (unstructured annotations)
        adata.uns["model_info"] = {
            "model_name": self.model.__class__.__name__,
            "device": str(self.device),
            "total_samples": len(results["input"]),
            "input_shape": results["input"][0].shape,
            "output_shape": results["target"][0].shape,
        }

        if self.logger is not None:
            self.logger.info(f"AnnData object created successfully.")
            self.logger.debug(adata)

        return adata
