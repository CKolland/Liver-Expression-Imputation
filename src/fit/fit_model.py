from datetime import datetime

import anndata as ad
import pandas as pd
import torch

from fit.model import MLP
import utils.confy as confy
import utils.io_utils as io
from utils.io_utils import ImputationDataset
from utils.train_utils import fit_kfold, save_training_results


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
    logger.info("Setup complete.")

    # -----------------
    #     Load data
    # -----------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Following device supported for this run: {device}.")

    run_setup = io.load_toml(setup_file)

    # Load training data
    train_data_config = confy.setup_dataset(run_setup["train_data"])
    train_data = ad.read_h5ad(train_data_config.path)
    logger.info(f"📁 Train data loaded succesfully: '{train_data_config.path}'.")
    logger.debug(train_data)

    # Chose data layer
    if train_data_config.layer is not None:
        train_data.X = train_data.layers[train_data_config.layer]
        logger.info(f"Selected {train_data_config.layer} as data layer.")

    # Load imputation mask
    masking_config = confy.setup_dataset(run_setup["imputation_mask"])
    imputation_mask = pd.read_csv(
        masking_config.path,
        header=None if masking_config.header == 0 else 0,
    )
    imputation_mask = imputation_mask[0].tolist()
    logger.info(f"📁 Imputation mask loaded succesfully: '{masking_config.path}'.")

    train_dataset = ImputationDataset(train_data, imputation_mask)
    logger.info("Created training dataset.")

    # -------------------
    #     Setup model
    # -------------------

    model_config = confy.setup_model(run_setup["model"])
    mlp = MLP(model_config.to_torch(), device)
    logger.info("Setup model successfully.")
    logger.debug(mlp)

    # -------------------
    #     Train model
    # -------------------

    train_config = confy.setup_training(run_setup["training"])

    # Train with KFold CV
    best_model, metrics = fit_kfold(
        model_template=mlp,
        dataset=train_dataset,
        config=train_config,
        device=device,
        logger=logger,
        seed=42,
    )

    # Save results
    torch.save(best_model.state_dict(), "best_model.pth")
    save_training_results(metrics, run_dir, logger)

    # Analyze results
    detailed_metrics = metrics.to_dataframe()
    fold_summary = metrics.get_fold_summary()
    print(f"Best fold: {metrics.best_fold + 1}")
    print(f"Best val loss: {metrics.best_val_loss}")
