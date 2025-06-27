from datetime import datetime

import anndata as ad
import pandas as pd
import torch

from fit.model import MLP
import utils.confy as confy
import utils.io_utils as io
from utils.io_utils import ImputationDataset, TrainingMetrics
from utils.train_utils import TrainingPipeline


def fit_model(path_to_setup: str, path_to_out: str):
    """_summary_

    :param str config_path: _description_
    :param str out_path: _description_
    """

    # Verify paths
    setup_file = io.assert_path(path_to_setup, assert_dir=False)  # File
    out_dir = io.assert_path(path_to_out)  # Directory

    # Create run directory
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = out_dir / f"training_run_{now}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------
    #     Setup logging
    # ---------------------

    logger = io.setup_logging(run_dir / f"training_run_{now}.log", "train_model")
    logger.info("✅ Setup complete.")

    # -----------------
    #     Load data
    # -----------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Supported device for this run: {device}.")
    logger.info("----")

    run_setup = io.load_toml(setup_file)

    # Load training data
    train_data_config = confy.setup_dataset(run_setup["train_data"])
    train_data = ad.read_h5ad(train_data_config.path)
    logger.info(f"📁 Training data loaded successfully: '{train_data_config.path}'.")

    # Choose data layer if provided
    if train_data_config.layer is not None:
        train_data.X = train_data.layers[train_data_config.layer]
        logger.info(f"Selected {train_data_config.layer} as data layer.")

    logger.debug(train_data)

    # Load imputation mask
    masking_config = confy.setup_dataset(run_setup["imputation_mask"])
    imputation_mask = pd.read_csv(masking_config.path, header=masking_config.header)
    imputation_mask = imputation_mask[0].tolist()
    logger.info(f"📁 Mask for imputation loaded successfully: '{masking_config.path}'.")

    train_dataset = ImputationDataset(train_data, imputation_mask)
    logger.info("Created training dataset.")
    logger.debug(train_dataset)
    logger.info("----")

    # -------------------
    #     Setup model
    # -------------------

    model_config = confy.setup_model(run_setup["model"])
    mlp = MLP(model_config.to_torch(), device)
    logger.info("Built model successfully.")
    logger.debug(mlp)
    logger.info("----")

    # -------------------
    #     Train model
    # -------------------

    train_config = confy.setup_training(run_setup["training"])

    pipeline = TrainingPipeline(
        train_dataset,
        mlp,
        device,
        logger,
        train_config.kfolds,
        train_config.epochs,
        train_config.early_stopping.patience,
        train_config.early_stopping.delta,
        train_config.batch_size,
        train_config.loss,
        train_config.optimization.get_optimizer(train_config.optimization.optimizer),
        train_config.optimization.learning_rate,
        train_config.optimization.weight_decay,
        train_config.seed,
        train_config.num_workers,
    )

    metrics = TrainingMetrics()
    best_model, metrics = pipeline.fit_kfold(metrics)
    pipeline.save_training_results(best_model, metrics, run_dir)

    logger.info(f"Best fold: {metrics.best_fold + 1}")
    logger.info(f"Best val loss: {metrics.best_val_loss}")
