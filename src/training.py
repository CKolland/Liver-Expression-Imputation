import copy
from dataclasses import dataclass
from logging import Logger

from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """_summary_"""

    def __init__(self, patience: int = 20, delta: float = 0):
        """_summary_

        :param int, optional patience: _description_, defaults to 20
        :param float, optional delta: _description_, defaults to 0
        """
        self.patience = patience
        self.delta = delta

        self.early_stop: bool = False
        self.counter: int = 0
        self.best_score: float = None
        self.best_model: dict = None

    def __call__(self, val_loss, model):
        """_summary_

        :param val_loss: _description_
        :type val_loss: _type_
        :param model: _description_
        :type model: _type_
        """
        score = -val_loss  # We negate because lower loss is better

        if self.best_score is None:  # Set initial best score
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:  # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # Improvement found
            self.counter = 0
            self.best_score = score
            self.best_model_state = model.state_dict()

    def load_model(self, model):
        """_summary_

        :param model: _description_
        :type model: _type_
        """
        model.load_state_dict(self.best_model)


@dataclass
class TrainingOutput:
    """Docstring"""

    train_loss: dict[float]
    val_loss: dict[float]

    def add_train_loss(self, epoch: int, loss: float):
        """_summary_

        :param epoch: _description_
        :type epoch: int
        :param loss: _description_
        :type loss: float
        """
        self.train_loss[str(epoch)] = loss

    def add_val_loss(self, epoch: int, loss: float):
        """_summary_

        :param epoch: _description_
        :type epoch: int
        :param loss: _description_
        :type loss: float
        """
        self.val_loss[str(epoch)] = loss


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epoch: int,
    train_output: TrainingOutput,
    device: str,
    logger: Logger,
):
    """Docstring"""
    # Set model to training mode
    model.train()

    # Initialize training loss for this epoch
    train_loss = 0.0
    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Training EPOCH {epoch + 1}",
    ):
        logger.info(f"Train EPOCH {epoch + 1}, BATCH {batch_idx + 1}")

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Clear gradients
        output = model(data)
        # Average batch loss stored in output
        loss = output.loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters

        # Sum average batch loss
        train_loss += loss

    # Store average training loss for epoch
    train_output.add_train_loss(epoch, train_loss / len(train_loader))

    return train_output


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    train_output: TrainingOutput,
    device: str,
    logger: Logger,
):
    """Docstring"""
    # Set model to validation mode
    model.eval()

    # Initialize validation loss for this epoch
    val_loss = 0.0
    with torch.no_grad():  # Prevents gradient calculation during evaluation
        for batch_idx, (data, target) in tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Validation EPOCH {epoch + 1}",
        ):
            logger.info(f"Validate EPOCH {epoch + 1}, BATCH {batch_idx + 1}")

            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = output.loss

            # Sum average batch loss
            val_loss += loss

    # Store average validation loss for epoch
    train_output.add_val_loss(epoch, val_loss / len(val_loader))

    return train_output


def test(model: nn.Module, test_loader: DataLoader, device: str, logger: Logger):
    """_summary_"""
    # Set model to validation mode
    model.eval()

    targets, predictions = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"Testing",
        ):
            data, target = data.to(device), target.to(device)

            output = model(data)

            targets.append(target.cpu().numpy())
            predictions.append(output.y.cpu().numpy())

    # Concatenate all batches
    targets = np.vstack(targets)  # Shape: (n_samples, n_genes)
    predictions = np.vstack(predictions)

    # Create AnnData to store results
    adata = AnnData(X=targets)
    adata.layers["predictions"] = predictions

    return adata


def fit(
    model,
    train_dataset,
    sc_test_loader,
    st_test_loader,
    t_conf,
    device: str,
    logger: Logger,
):
    """_summary_"""

    # Initialize the k-fold cross validation
    kf = KFold(n_splits=t_conf.kfolds, shuffle=True)

    best_val_loss = float("inf")
    best_model_state = None

    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
        logger.info(f"---- Fold {fold + 1} ----")

        # Create fresh model and optimizer for each fold
        fold_model = copy.deepcopy(model)
        optimizer = t_conf.optimizer(fold_model.parameters(), lr=0.01)

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=t_conf.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        val_loader = DataLoader(
            dataset=train_dataset,
            batch_size=t_conf.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        # Setup early stopping
        early_stopping = EarlyStopping()
        train_output = TrainingOutput()

        # Train the model on the current fold
        for epoch in range(t_conf.epochs):
            train_output = train(
                fold_model, optimizer, train_loader, epoch, train_output, device, logger
            )
            train_output = validate(
                fold_model, val_loader, epoch, train_output, device, logger
            )

            train_loss, val_loss = (
                train_output.train_loss[str(epoch)],
                train_output.val_loss[str(epoch)],
            )

            logger.info(
                f"Average TRAIN loss: {train_loss}, average VAL loss: {val_loss}"
            )

            early_stopping(val_loss, fold_model)
            if early_stopping.early_stop:
                logger.info(f"EARLY STOPPING after {epoch} epochs")
                break

        # Compare current model to best so far
        if early_stopping.best_score < best_val_loss:
            best_val_loss = early_stopping.best_score
            best_model = copy.deepcopy(early_stopping.best_model.state_dict())
            best_training = train_output
            logger.info(
                f"Best model updated (Fold {fold + 1}, Val Loss: {best_val_loss})"
            )

    # Convert training to data frame
    train_results = pd.DataFrame(
        {
            "epoch": list(train_output.train_loss.keys()),
            "training_loss": list(train_output.train_loss.values()),
            "validation_loss": list(train_output.val_loss.values()),
        }
    )

    # Load best model for final testing
    model.load_state_dict(best_model_state)
    scdata = test(model, sc_test_loader, device, logger)
    stdata = test(model, st_test_loader, device, logger)

    return best_model, train_results, scdata, stdata
