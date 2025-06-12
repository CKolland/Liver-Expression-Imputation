import argparse
from datetime import datetime
import os
import os.path as path

import anndata as ad
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_processing import ImputationDataset
from model import MLP
from training import fit
import utils.configurator as config
from utils.misc import setup_logging, verify_path


def main():
    """Docstring"""

    # --------------------
    #     Setup parser
    # --------------------

    parser = argparse.ArgumentParser(
        description="Single-cell count imputation for spatial transcriptomics of liver tissue"
    )

    parser.add_argument(
        "--config-path",
        "-c",
        type=str,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "--layer",
        "-l",
        type=str,
        default="log1p",
        help="Name of the layer that contains the scaled data",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        required=True,
        help="Path to the output directory",
    )

    args = parser.parse_args()

    # Verify paths
    config_path = verify_path(args.config_path, is_dir=False)
    out_dir = verify_path(args.out_dir, is_dir=True)

    # Create run directory
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = path.join(out_dir, f"run_{now}")
    os.makedirs(run_dir, exist_ok=True)  # Create run directory

    # ---------------------
    #     Setup logging
    # ---------------------

    logger = setup_logging(os.path.join(run_dir, "run.log"))
    logger.info("Setup complete.")

    # -----------------
    #     Load data
    # -----------------

    d_conf, m_conf, t_conf = config.setup_classes(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and test data
    train_data = ad.read_h5ad(d_conf.trainig_data)
    train_data.X = train_data.layers[args.layer]
    logger.info(f"Train data loaded succesfully: {d_conf.trainig_data}")
    logger.debug(train_data)

    st_test_data = ad.read_h5ad(d_conf.test_data)
    st_test_data.X = st_test_data.layers[args.layer]
    logger.info(f"ST test data loaded succesfully: {d_conf.test_data}")
    logger.debug(st_test_data)

    # Load list of genes that define which genes are not masked
    imputation_mask = pd.read_csv(d_conf.imputation_mask)

    # Calculate the number of entries for training and test data
    train_size = int(0.9 * train_data.shape[0])

    # Set seed to make data distribution reproducable
    torch.manual_seed(12062025)
    perm = torch.randperm(train_data.shape[0])
    train_split, test_split = perm[:train_size], perm[train_size:]

    if logger is not None:
        logger.info(f"Training split created. Contains {len(train_split)} entries")
        logger.info(f"Validation split created. Contains {len(test_split)} entries")

    train_dataset = ImputationDataset(
        train_data[train_split.numpy(), :], imputation_mask
    )
    sc_test_dataset = ImputationDataset(
        train_data[test_split.numpy(), :], imputation_mask
    )
    st_test_dataset = ImputationDataset(st_test_data, imputation_mask)

    # Create test dataloaders
    sc_test_loader = DataLoader(
        sc_test_dataset,
        batch_size=t_conf.batch_size,
        shuffle=False,
    )
    st_test_loader = DataLoader(
        st_test_dataset,
        batch_size=t_conf.batch_size,
        shuffle=False,
    )

    # -------------------
    #     Setup model
    # -------------------

    mlp = MLP(m_conf.to_torch(), m_conf.loss())
    logger.info("Setup model")
    logger.debug(mlp)

    # -------------------
    #     Train model
    # -------------------

    best_model, train_results, scdata, stdata = fit(
        mlp,
        train_dataset,
        sc_test_loader,
        st_test_loader,
        t_conf,
        device,
        logger,
    )

    # ---------------------
    #     Store results
    # ---------------------

    scdata.write_h5ad(os.path.join(out_dir, "sc_test_results.h5ad"))
    stdata.write_h5ad(os.path.join(out_dir, "st_test_results.h5ad"))
    torch.save(best_model, os.path.join(out_dir, "imputation_model.pt"))
    train_results.to_feather(os.path.join(out_dir, "training_results.feather"))

    logger.info("FINISHED!")


if __name__ == "__main__":
    main()
