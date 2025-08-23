import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from utils.io import assert_path
from utils.vis import (
    apply_threshold,
    calc_test_metrics,
    plot_frequency,
    plot_targets_vs_predictions,
)
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
            mask_names = masks[masks[mask]].index.to_list()
            # Returns tuple therefor 0 index has to be accessed
            target_indices = np.where(np.isin(adata.uns["target_names"], mask_names))[0]
            targets = adata.obsm["targets"][:, target_indices]
            predictions = adata.obsm["predictions"][:, masks[mask].to_numpy()]
            gene_names = masks[masks[mask]].index.to_list()

            gene_wise_metrics, _ = calc_test_metrics(targets, predictions, gene_names)
            gene_wise_metrics.to_feather(
                out_dir / f"{mask}_{adata_file.stem}_gene_wise_metrics.feather"
            )
            logging.info(f"Saved metrics for mask '{mask}'.")

            # Histogram Pearson correlation
            file_name = out_dir / f"{mask}_pearson_frequency.png"

            p = plot_frequency(
                gene_wise_metrics,
                "pearson_correlation",
                title="Frequency Pearson Correlation",
                x_lab="Pearson Correlation",
                save_to=file_name,
            )

            # Histogram Spearman correlation
            file_name = out_dir / f"{mask}_spearman_frequency.png"

            p = plot_frequency(
                gene_wise_metrics,
                "spearman_correlation",
                title="Frequency Spearman Correlation",
                x_lab="Spearman Correlation",
                save_to=file_name,
            )

            # Mean expression target vs. mean expression predicted
            plot_data = gene_wise_metrics.rename(
                columns={
                    "target_mean": "targets",
                    "predicted_mean": "predictions",
                }
            )[["targets", "predictions"]]

            file_name = out_dir / f"{mask}_target_expr_vs_pred_expr.png"

            p = plot_targets_vs_predictions(
                plot_data,
                title="Target vs. Predicted Expression",
                x_lab="Mean Expression Targets",
                y_lab="Mean Expression Predictions",
                add_diag=True,
                save_to=file_name,
            )

            # Histogram MAE
            file_name = out_dir / f"{mask}_mae_frequency.png"

            p = plot_frequency(
                gene_wise_metrics,
                "mae",
                title="Frequency MAE",
                x_lab="MAE",
                save_to=file_name,
            )

            # MAE vs. mean expression target
            plot_data = gene_wise_metrics.rename(
                columns={
                    "mae": "targets",
                    "target_mean": "predictions",
                }
            )[["targets", "predictions"]]

            file_name = out_dir / f"{mask}_mae_vs_target_expr.png"

            p = plot_targets_vs_predictions(
                plot_data,
                title="MAE vs. Target Expression",
                x_lab="MAE",
                y_lab="Mean Expression Targets",
                add_diag=False,
                save_to=file_name,
            )

            # Target sparsity vs. predicted sparsity
            plot_data = gene_wise_metrics.rename(
                columns={
                    "target_sparsity": "targets",
                    "predicted_sparsity": "predictions",
                }
            )[["targets", "predictions"]]

            file_name = out_dir / f"{mask}_target_vs_pred_sparsity.png"

            p = plot_targets_vs_predictions(
                plot_data,
                title="Target vs. Predicted Sparsity",
                x_lab="Target Sparsity",
                y_lab="Predicted Sparsity",
                add_diag=True,
                save_to=file_name,
            )

            # Pearson correlation vs. target sparsity
            plot_data = gene_wise_metrics.rename(
                columns={
                    "pearson_correlation": "targets",
                    "target_sparsity": "predictions",
                }
            )[["targets", "predictions"]]

            file_name = out_dir / f"{mask}_pearson_vs_target_sparsity.png"

            p = plot_targets_vs_predictions(
                plot_data,
                title="Pearson Correlation vs. Target Sparsity",
                x_lab="Pearson Correlation",
                y_lab="Target Sparsity",
                add_diag=False,
                save_to=file_name,
            )

            # Spearman correlation vs. target sparsity
            plot_data = gene_wise_metrics.rename(
                columns={
                    "spearman_correlation": "targets",
                    "target_sparsity": "predictions",
                }
            )[["targets", "predictions"]]

            file_name = out_dir / f"{mask}_spearman_vs_target_sparsity.png"

            p = plot_targets_vs_predictions(
                plot_data,
                title="Spearman Correlation vs. Target Sparsity",
                x_lab="Spearman Correlation",
                y_lab="Target Sparsity",
                add_diag=False,
                save_to=file_name,
            )

            # RMSE vs. mean epxression target
            plot_data = gene_wise_metrics.rename(
                columns={
                    "rmse": "targets",
                    "target_mean": "predictions",
                }
            )[["targets", "predictions"]]

            file_name = out_dir / f"{mask}_rmse_vs_target_expr.png"

            p = plot_targets_vs_predictions(
                plot_data,
                title="RMSE vs. Mean Expression Target",
                x_lab="RMSE",
                y_lab="Mean Expressoin Target",
                add_diag=False,
                save_to=file_name,
            )

            logging.info(f"💾 Saved plots for mask '{mask}'.")
    else:
        targets = adata.obsm["targets"]
        predictions = adata.obsm["predictions"]
        gene_names = adata.var_names.to_list()

        gene_wise_metrics, _ = calc_test_metrics(targets, predictions, gene_names)
        gene_wise_metrics.to_feather(out_dir / f"{adata_file.stem}_metrics.feather")
        logging.info(f"Saved metrics.")

        # Histogram Pearson correlation
        file_name = out_dir / "pearson_frequency.png"

        p = plot_frequency(
            gene_wise_metrics,
            "pearson_correlation",
            title="Frequency Pearson Correlation",
            x_lab="Pearson Correlation",
            save_to=file_name,
        )

        # Histogram Spearman correlation
        file_name = out_dir / "spearman_frequency.png"

        p = plot_frequency(
            gene_wise_metrics,
            "spearman_correlation",
            title="Frequency Spearman Correlation",
            x_lab="Spearman Correlation",
            save_to=file_name,
        )

        # Mean expression target vs. mean expression predicted
        plot_data = gene_wise_metrics.rename(
            columns={
                "target_mean": "targets",
                "predicted_mean": "predictions",
            }
        )[["targets", "predictions"]]

        file_name = out_dir / "target_expr_vs_pred_expr.png"

        p = plot_targets_vs_predictions(
            plot_data,
            title="Target vs. Predicted Expression",
            x_lab="Mean Expression Targets",
            y_lab="Mean Expression Predictions",
            add_diag=True,
            save_to=file_name,
        )

        # Histogram MAE
        file_name = out_dir / "mae_frequency.png"

        p = plot_frequency(
            gene_wise_metrics,
            "mae",
            title="Frequency MAE",
            x_lab="MAE",
            save_to=file_name,
        )

        # MAE vs. mean expression target
        plot_data = gene_wise_metrics.rename(
            columns={
                "mae": "targets",
                "target_mean": "predictions",
            }
        )[["targets", "predictions"]]

        file_name = out_dir / "mae_vs_target_expr.png"

        p = plot_targets_vs_predictions(
            plot_data,
            title="MAE vs. Target Expression",
            x_lab="MAE",
            y_lab="Mean Expression Targets",
            add_diag=False,
            save_to=file_name,
        )

        # Target sparsity vs. predicted sparsity
        plot_data = gene_wise_metrics.rename(
            columns={
                "target_sparsity": "targets",
                "predicted_sparsity": "predictions",
            }
        )[["targets", "predictions"]]

        file_name = out_dir / "target_vs_pred_sparsity.png"

        p = plot_targets_vs_predictions(
            plot_data,
            title="Target vs. Predicted Sparsity",
            x_lab="Target Sparsity",
            y_lab="Predicted Sparsity",
            add_diag=True,
            save_to=file_name,
        )

        # Pearson correlation vs. target sparsity
        plot_data = gene_wise_metrics.rename(
            columns={
                "pearson_correlation": "targets",
                "target_sparsity": "predictions",
            }
        )[["targets", "predictions"]]

        file_name = out_dir / "pearson_vs_target_sparsity.png"

        p = plot_targets_vs_predictions(
            plot_data,
            title="Pearson Correlation vs. Target Sparsity",
            x_lab="Pearson Correlation",
            y_lab="Target Sparsity",
            add_diag=False,
            save_to=file_name,
        )

        # Spearman correlation vs. target sparsity
        plot_data = gene_wise_metrics.rename(
            columns={
                "spearman_correlation": "targets",
                "target_sparsity": "predictions",
            }
        )[["targets", "predictions"]]

        file_name = out_dir / "spearman_vs_target_sparsity.png"

        p = plot_targets_vs_predictions(
            plot_data,
            title="Spearman Correlation vs. Target Sparsity",
            x_lab="Spearman Correlation",
            y_lab="Target Sparsity",
            add_diag=False,
            save_to=file_name,
        )

        # RMSE vs. mean epxression target
        plot_data = gene_wise_metrics.rename(
            columns={
                "rmse": "targets",
                "target_mean": "predictions",
            }
        )[["targets", "predictions"]]

        file_name = out_dir / "rmse_vs_target_expr.png"

        p = plot_targets_vs_predictions(
            plot_data,
            title="RMSE vs. Mean Expression Target",
            x_lab="RMSE",
            y_lab="Mean Expressoin Target",
            add_diag=False,
            save_to=file_name,
        )

        logging.info(f"💾 Saved plots.")


def compute_metrics_on_threshold(
    path_to_adata: str,
    path_to_masks: str,
    threshold: float,
):
    """_summary_

    :param path_to_adata: _description_
    :type path_to_adata: str
    :param path_to_masks: _description_
    :type path_to_masks: str
    :param threshold: _description_
    :type threshold: float
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

    out_dir = adata_file.parent / f"{adata_file.stem}_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"All metrics are saved in '{out_dir}'.")

    masks = pd.read_feather(path_to_masks)

    for mask in masks:
        mask_names = masks[masks[mask]].index.to_list()
        # Returns tuple therefor 0 index has to be accessed
        target_indices = np.where(np.isin(adata.uns["target_names"], mask_names))[0]
        targets = adata.obsm["targets"][:, target_indices]
        predictions = adata.obsm["predictions"][:, masks[mask].to_numpy()]
        gene_names = masks[masks[mask]].index.to_list()

        # Apply threshold to predictions
        predictions = apply_threshold(predictions, threshold=threshold)

        gene_wise_metrics, _ = calc_test_metrics(targets, predictions, gene_names)
        gene_wise_metrics.to_feather(
            out_dir / f"{mask}_{adata_file.stem}_threshold_gene_metrics.feather"
        )
