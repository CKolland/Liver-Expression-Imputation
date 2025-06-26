import copy
from logging import Logger
from typing import Any, Optional

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from utils.confy import TrainingConfig
from utils.io_utils import ImputationDataset, TrainingMetrics


class EarlyStopping:
    """Early stopping utility to prevent overfitting during model training.

    This class monitors validation loss and stops training when the loss stops
    improving for a specified number of epochs (patience). It also saves the
    best model state for later restoration.
    """

    def __init__(self, patience: int = 20, delta: float = 0.0):
        """Initialize the EarlyStopping callback.

        :param int patience: Number of epochs with no improvement after which training will be stopped, defaults to 20
        :param float delta: Minimum change in the monitored quantity to qualify as an improvement, defaults to 0.0
        """
        self.patience: int = patience
        self.delta: float = delta
        self.early_stop: bool = False
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_model_state: Optional[dict[str, Any]] = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Check if early stopping criteria are met and update best model if improved.

        :param float val_loss: Current validation loss
        :param nn.Module model: PyTorch model to monitor and potentially save
        :raises TypeError: If val_loss is not a number or model is not a PyTorch module
        """
        # Input validation
        if not isinstance(val_loss, (int, float)):
            raise TypeError("val_loss must be a numeric value")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a PyTorch nn.Module")

        # Convert validation loss to score (negative because lower loss is better)
        score = -val_loss

        if self.best_score is None:
            # First epoch - initialize best score and save model state
            self.best_score = score
            self.best_model_state = model.state_dict().copy()  # Use copy() for safety
        elif score < self.best_score + self.delta:
            # No significant improvement detected
            self.counter += 1

            if self.counter >= self.patience:
                # Patience exhausted - trigger early stopping
                self.early_stop = True
        else:
            # Improvement detected - reset counter and save new best model
            self.counter = 0
            self.best_score = score
            self.best_model_state = model.state_dict().copy()  # Use copy() for safety

    def load_best_model(self, model: nn.Module) -> None:
        """Load the best model state back into the provided model.

        :param nn.Module model: PyTorch model to load the best state into
        :raises RuntimeError: If no best model state has been saved yet
        :raises TypeError: If model is not a PyTorch module
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a PyTorch nn.Module")

        if self.best_model_state is None:
            raise RuntimeError(
                "No best model state available. Call this method only after training has started."
            )

        # Load the best model state dictionary
        model.load_state_dict(self.best_model_state)

    def reset(self) -> None:
        """Reset the early stopping state for a new training session.

        This method clears all internal state, allowing the EarlyStopping
        instance to be reused for multiple training runs.
        """
        self.early_stop = False
        self.counter = 0
        self.best_score = None
        self.best_model_state = None


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    device: str,
    epoch: int,
    fold: int,
) -> float:
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Fold {fold+1} - Train Epoch {epoch+1}")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target).sum(-1).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / len(train_loader)


def validate_epoch(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: str,
    epoch: int,
    fold: int,
) -> float:
    """Validate model for one epoch and return average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Fold {fold+1} - Val Epoch {epoch+1}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target).sum(-1).mean()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / len(val_loader)


def fit_kfold(
    model_template: nn.Module,
    dataset: ImputationDataset,
    config: TrainingConfig,
    device: str,
    logger: Logger,
    seed: int = 42,
) -> tuple[nn.Module, TrainingMetrics]:
    """
    Train model using KFold cross-validation with comprehensive tracking.

    Args:
        model_template: Template model to be copied for each fold
        dataset: Training dataset
        config: Training configuration object
        device: Device to train on
        logger: Logger instance
        seed: Random seed for reproducibility

    Returns:
        Tuple of (best_model, training_metrics)
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize KFold with seed for reproducibility
    kfold = KFold(n_splits=config.kfolds, shuffle=True, random_state=seed)
    metrics = TrainingMetrics()

    logger.info(f"Starting {config.kfolds}-fold cross-validation with seed {seed}...")

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"----")
        logger.info(f"FOLD {fold + 1}/{config.kfolds}")
        logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        # Create fresh model and optimizer for each fold
        fold_model = copy.deepcopy(model_template)
        fold_model.to(device)
        optimizer = config.optimization.create_optimizer(fold_model.parameters())

        # Create data loaders for current fold
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=getattr(config, "num_workers", 0),
        )
        val_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            sampler=SubsetRandomSampler(val_idx),
            num_workers=getattr(config, "num_workers", 0),
        )

        # Initialize early stopping for this fold
        early_stopping = EarlyStopping(patience=getattr(config, "patience", 20))

        # Training loop for current fold
        for epoch in range(config.epochs):
            # Train and validate
            train_loss = train_epoch(
                fold_model, optimizer, config.loss(), train_loader, device, epoch, fold
            )
            val_loss = validate_epoch(
                fold_model, config.loss(), val_loader, device, epoch, fold
            )

            # Record metrics
            metrics.add_fold_epoch(fold, epoch, train_loss, val_loss)

            logger.info(
                f"Fold {fold+1}, Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Check early stopping
            early_stopping(val_loss, fold_model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Update best model if this fold performed better
        fold_best_val_loss = (
            -early_stopping.best_score if early_stopping.best_score else val_loss
        )
        metrics.update_best_model(
            fold, fold_best_val_loss, early_stopping.best_model_state
        )

        logger.info(f"Fold {fold+1} completed. Best val loss: {fold_best_val_loss:.6f}")

    # Create final best model
    best_model = copy.deepcopy(model_template)
    best_model.load_state_dict(metrics.best_model_state)
    best_model.to(device)

    logger.info(f"=" * 50)
    logger.info(f"Cross-validation completed!")
    logger.info(
        f"Best model from fold {metrics.best_fold + 1} with val loss: {metrics.best_val_loss:.6f}"
    )

    return best_model, metrics


def save_training_results(
    metrics: TrainingMetrics, save_path: str, logger: Logger
) -> None:
    """Save training results to files."""
    try:
        # Save detailed metrics
        detailed_df = metrics.to_dataframe()
        detailed_df.to_csv(f"{save_path}_detailed_metrics.csv", index=False)

        # Save fold summary
        summary_df = metrics.get_fold_summary()
        summary_df.to_csv(f"{save_path}_fold_summary.csv", index=False)

        logger.info(f"Training results saved to {save_path}_*.csv")

    except Exception as e:
        logger.error(f"Failed to save training results: {e}")


def load_best_model(
    model_template: nn.Module, model_path: str, device: str
) -> nn.Module:
    """Load the best model from a saved state dict."""
    model = copy.deepcopy(model_template)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# def train(
#     model: nn.Module,
#     optimizer: optim.Optimizer,
#     criterion: nn.Module,
#     train_loader: DataLoader,
#     epoch: int,
#     train_output,
#     device: str,
#     logger: Logger,
# ):
#     # Set model to training mode
#     model.train()

#     # Initialize training loss for this epoch
#     train_loss = 0.0
#     for batch_idx, (data, target) in tqdm(
#         enumerate(train_loader),
#         total=len(train_loader),
#         desc=f"Training EPOCH {epoch + 1}",
#     ):
#         data, target = data.to(device), target.to(device)

#         optimizer.zero_grad()  # Clear gradients
#         output = model(data)
#         # `loss_function(z, x)` computes tensor of shape [z, x] with element-wise loss
#         # `.sum(-1)` sums loss over predicted features (total loss per sample)
#         # `.mean()` averages the per sample loss over batch
#         loss = criterion(output.y, target).sum(-1).mean()
#         loss.backward()  # Backpropagation
#         optimizer.step()  # Update parameters

#         # Sum average batch loss
#         train_loss += loss

#     # Store average training loss for epoch
#     train_output.add_train_loss(epoch, train_loss / len(train_loader))

#     return train_output


# def validate(
#     model: nn.Module,
#     criterion: nn.Module,
#     val_loader: DataLoader,
#     epoch: int,
#     train_output,
#     device: str,
#     logger: Logger,
# ):
#     """Docstring"""
#     # Set model to validation mode
#     model.eval()

#     # Initialize validation loss for this epoch
#     val_loss = 0.0
#     with torch.no_grad():  # Prevents gradient calculation during evaluation
#         for batch_idx, (data, target) in tqdm(
#             enumerate(val_loader),
#             total=len(val_loader),
#             desc=f"Validation EPOCH {epoch + 1}",
#         ):
#             data, target = data.to(device), target.to(device)

#             output = model(data)
#             loss = criterion(output.y, target).sum(-1).mean()

#             # Sum average batch loss
#             val_loss += loss

#     # Store average validation loss for epoch
#     train_output.add_val_loss(epoch, val_loss / len(val_loader))

#     return train_output


# def test(model: nn.Module, test_loader: DataLoader, device: str, logger: Logger):
#     """_summary_"""
#     # Set model to validation mode
#     model.eval()

#     targets, predictions = [], []
#     with torch.no_grad():
#         for batch_idx, (data, target) in tqdm(
#             enumerate(test_loader),
#             total=len(test_loader),
#             desc=f"Testing",
#         ):
#             data, target = data.to(device), target.to(device)

#             output = model(data)

#             targets.append(target.cpu().numpy())
#             predictions.append(output.y.cpu().numpy())

#     # Concatenate all batches
#     targets = np.vstack(targets)  # Shape: (n_samples, n_genes)
#     predictions = np.vstack(predictions)

#     # Create AnnData to store results
#     adata = ad.AnnData(X=targets)
#     adata.layers["predictions"] = predictions

#     return adata


# def fit(
#     model: nn.Module,
#     train_dataset: ImputationDataset,
#     config,
#     device: str,
#     logger: Logger,
# ):
#     """_summary_"""

#     # Initialize the k-fold cross validation
#     kf = KFold(n_splits=config.kfolds, shuffle=True)

#     best_val_loss = float("inf")
#     best_model_state = None

#     # Loop through each fold
#     for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
#         logger.info(f"---- Fold {fold + 1} ----")

#         # Create fresh model and optimizer for each fold
#         fold_model = copy.deepcopy(model)
#         optimizer = t_conf.optimizer(fold_model.parameters(), lr=0.01)

#         # Define the data loaders for the current fold
#         train_loader = DataLoader(
#             dataset=train_dataset,
#             batch_size=t_conf.batch_size,
#             sampler=torch.utils.data.SubsetRandomSampler(train_idx),
#         )
#         val_loader = DataLoader(
#             dataset=train_dataset,
#             batch_size=t_conf.batch_size,
#             sampler=torch.utils.data.SubsetRandomSampler(test_idx),
#         )

#         # Setup early stopping
#         early_stopping = EarlyStopping()

#         # Train the model on the current fold
#         for epoch in range(t_conf.epochs):
#             train_output = train(
#                 fold_model,
#                 optimizer,
#                 t_conf.loss(),
#                 train_loader,
#                 epoch,
#                 train_output,
#                 device,
#                 logger,
#             )
#             train_output = validate(
#                 fold_model,
#                 t_conf.loss(),
#                 val_loader,
#                 epoch,
#                 train_output,
#                 device,
#                 logger,
#             )

#             train_loss, val_loss = (
#                 train_output.train_loss[str(epoch)],
#                 train_output.val_loss[str(epoch)],
#             )

#             logger.info(
#                 f"Average TRAIN loss: {train_loss}, average VAL loss: {val_loss}"
#             )

#             early_stopping(val_loss, fold_model)
#             if early_stopping.early_stop:
#                 logger.info(f"EARLY STOPPING after {epoch} epochs")
#                 break

#         # Compare current model to best so far
#         if early_stopping.best_score < best_val_loss:
#             best_val_loss = early_stopping.best_score
#             best_model = copy.deepcopy(early_stopping.best_model.state_dict())
#             best_training = train_output
#             logger.info(
#                 f"Best model updated (Fold {fold + 1}, Val Loss: {best_val_loss})"
#             )

#     # Convert training to data frame
#     train_results = pd.DataFrame(
#         {
#             "epoch": list(train_output.train_loss.keys()),
#             "training_loss": list(train_output.train_loss.values()),
#             "validation_loss": list(train_output.val_loss.values()),
#         }
#     )

#     # Load best model for final testing
#     model.load_state_dict(best_model_state)
#     scdata = test(model, sc_test_loader, device, logger)
#     stdata = test(model, st_test_loader, device, logger)

#     return best_model, train_results, scdata, stdata
