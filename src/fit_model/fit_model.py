from datetime import datetime, timedelta
import logging
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim

from utils.model import MLP
import utils.confy as confy
import utils.io as io
from utils.io import ImputationDataset, TrainingMetrics
from utils.fit import TrainingPipeline

import fit_model._constants as C


def test_model_fitting():
    """_summary_"""
    n_features = 1000
    n_samples = 10000
    seed = 30062025

    # ---------------------
    #     Setup logging
    # ---------------------

    # Setup custom logging
    logger = logging.getLogger("test_model_fitting")
    logger.setLevel(logging.INFO)  # Lowest log level (logs everything)

    # Custom formatter
    formatter = logging.Formatter(C.LOGGING_FORMAT, datefmt=C.LOGGING_DATEFMT)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Set handler specific log level
    console_handler.setFormatter(formatter)  # Add custom formatter to handler
    logger.addHandler(console_handler)  # Add handler to the logger
    logger.info("✅ Setup complete.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Supported device for this run: {device}.")
    logger.info("----")

    # ---------------------
    #     Generate data
    # ---------------------

    logger.info("Start TEST run")
    logger.info("----")

    rng = np.random.default_rng(seed)

    # Simulate expression counts (Poisson-distributed)
    # Base expression levels per gene
    base_expression = rng.uniform(0.1, 5.0, size=n_features)
    # Generate cell x gene count matrix
    expression_matrix = rng.poisson(lam=base_expression, size=(n_samples, n_features))

    # Convert to sparse matrix
    X = csr_matrix(expression_matrix)

    # Create cell metadata (obs)
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_samples)],
            "sample": rng.choice(["sample_1", "sample_2"], size=n_samples),
        }
    ).set_index("cell_id")

    # Create gene metadata (var)
    var = pd.DataFrame(
        {
            "gene_id": [f"gene_{i}" for i in range(n_features)],
            "highly_variable": rng.choice([True, False], size=n_features, p=[0.2, 0.8]),
        }
    ).set_index("gene_id")

    # Construct AnnData
    syn_data = ad.AnnData(X=X, obs=obs, var=var)
    logger.info(f"Generated synthetic data.\n{syn_data}")

    # Choose random genes
    np.random.seed(seed)
    selected_genes = np.random.choice(n_features, size=100, replace=False)
    genes_to_exclude = syn_data.var_names[selected_genes]
    testing_mask = ~syn_data.var_names.isin(genes_to_exclude)
    imputation_mask = syn_data.var_names[testing_mask]

    train_dataset = ImputationDataset(syn_data, imputation_mask)
    logger.info(f"Created training dataset.\n{len(train_dataset)}")
    logger.info("----")

    # -------------------
    #     Setup model
    # -------------------

    mlp = MLP(
        nn.Sequential(
            nn.Linear(900, 950),
            nn.ReLU(),
            nn.Linear(950, 1000),
        ),
        device,
    )
    logger.info(f"Model built successfully.\n{mlp}")
    logger.info("----")

    # -------------------
    #     Train model
    # -------------------

    pipeline = TrainingPipeline(
        train_dataset,
        mlp,
        device,
        logger,
        kfolds=2,
        epochs=5,
        batch_size=64,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=seed,
        num_workers=8,
    )

    metrics = TrainingMetrics()

    start_time = time.time()
    _, metrics = pipeline.fit_kfold(metrics)
    end_time = time.time()

    elapsed_time = end_time - start_time
    formatted_time = str(timedelta(seconds=int(elapsed_time)))
    logger.info(f"Model fitting took: {formatted_time} (HH:MM:SS)")


def fit_model(path_to_setup: str, path_to_out: str):
    """_summary_

    :param str config_path: _description_
    :param str out_path: _description_
    """
    # Verify paths
    setup_file = io.assert_path(path_to_setup, assert_dir=False)  # File
    out_dir = io.assert_path(path_to_out)  # Directory
    
    # Load model config here to get model name
    run_setup = io.load_toml(setup_file)
    model_config = confy.setup_model(run_setup["model"])

    # Create run directory
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = out_dir / f"{model_config.name}_run_{now}"
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
        train_config.loss(),
        train_config.optimization.get_optimizer(),
        train_config.optimization.learning_rate,
        train_config.optimization.weight_decay,
        train_config.optimization.use_scheduler,
        train_config.seed,
        train_config.num_workers,
    )

    metrics = TrainingMetrics()
    best_model, metrics = pipeline.fit_kfold(metrics)
    pipeline.save_training_results(best_model, metrics, run_dir)
