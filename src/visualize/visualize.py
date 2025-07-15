import logging

import anndata as ad
import pandas as pd

from utils.io import assert_path
from utils.vis import plot_target_vs_prediction
import visualize._constants as C


def visualize_metrics():
    """_summary_"""
    pass


def visualize_test(path_to_adata: str, custom_masks: str | None):
    """_summary_

    :param path_to_adata: _description_
    :type path_to_adata: str
    """
    # Setup custom logging
    logging.basicConfig(
        level=getattr(logging, C.LOGGING_LVL_CONSOLE),
        format=C.LOGGING_FORMAT,
        datefmt=C.LOGGING_DATEFMT,
    )

    logging.info("✅ Setup complete.")
    logging.info("----")

    adata_file = assert_path(path_to_adata, assert_dir=False)
    adata = ad.read_h5ad(adata_file)
    logging.info(f"AnnData object loaded successfully.\n{adata}")

    # Figures are saved in folder that is created in directory of the dataset
    # Each folder is created with the schematic: dataset name + "_figures"
    out_dir = adata_file.parent / f"{adata_file.stem}_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"All plots are saved in '{out_dir}'.")

    if custom_masks is not None:
        masks = pd.read_feather(custom_masks)

        for mask in masks:
            targets = adata.obsm["targets"][:, masks[mask].to_numpy()]
            predictions = adata.obsm["predictions"][:, masks[mask].to_numpy()]

            labels = ("Mean Expression Targets", "Mean Expression Predictions")
            save_file = out_dir / f"{mask}_target_vs_pred.png"

            plot_target_vs_prediction(
                targets,
                predictions,
                labels=labels,
                save_file=save_file,
            )
            logging.info(f"💾 Saved plot '{mask}_target_vs_pred.png'.")
    else:
        targets = adata.obsm["targets"]
        predictions = adata.obsm["predictions"]

        save_file = out_dir / f"target_vs_pred.png"

        plot_target_vs_prediction(targets, predictions, save_file=save_file)
        logging.info(f"💾 Save plotd 'target_vs_pred.png'.")
