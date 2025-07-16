import re
import warnings

import numpy as np
import pandas as pd
from plotnine import *
from plotnine.exceptions import PlotnineWarning
import scipy.sparse as sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

import utils._constants as C

# Suppress Plotnine warnings
warnings.filterwarnings("ignore", category=PlotnineWarning)


def _calc_axis_wise_metrics(
    targets: sparse.spmatrix,
    predictions: sparse.spmatrix,
    item_names: list[str],
    axis: int = C.GENE_AXIS,
    label: str = C.GENE_LABEL,
) -> pd.DataFrame:
    """Calculate axis-wise metrics for genes (axis=0) or cells (axis=1).

    This function computes various evaluation metrics (correlation, error measures,
    and sparsity statistics) for each gene or cell individually, depending on the
    specified axis.

    :param targets: True expression values as sparse matrix
    :type targets: sparse.spmatrix
    :param predictions: Predicted expression values as sparse matrix
    :type predictions: sparse.spmatrix
    :param item_names: Names of genes or cells corresponding to the axis
    :type item_names: list[str]
    :param axis: Axis along which to compute metrics (0 for genes, 1 for cells)
    :type axis: int
    :param label: Label for the DataFrame index column
    :type label: str

    :return: DataFrame containing per-item metrics
    :rtype: pd.DataFrame

    :raises ValueError: If axis is not 0 or 1
    """
    # Convert to appropriate sparse format for efficient column/row-wise operations
    if axis == C.GENE_AXIS:
        targets = targets.tocsc()
        predictions = predictions.tocsc()

        n_items = targets.shape[1]

        slice_func = lambda mat, i: mat[:, i]
    elif axis == C.CELL_AXIS:
        targets = targets.tocsr()
        predictions = predictions.tocsr()

        n_items = targets.shape[0]

        slice_func = lambda mat, i: mat[i, :]
    else:
        raise ValueError("Axis must be 0 (genes) or 1 (cells)")

    pearson_corrs, spearman_corrs = np.zeros(n_items), np.zeros(n_items)
    mse_vals, mae_vals = np.zeros(n_items), np.zeros(n_items)

    # Compute mean values across the specified axis
    target_means = np.array(targets.mean(axis=axis)).ravel()
    pred_means = np.array(predictions.mean(axis=axis)).ravel()

    # Calculate sparsity (proportion of zero values)
    target_sparsity = 1 - np.array(
        [slice_func(targets, i).nnz / targets.shape[axis] for i in range(n_items)]
    )
    pred_sparsity = 1 - np.array(
        [slice_func(targets, i).nnz / targets.shape[axis] for i in range(n_items)]
    )

    # Compute metrics for each item (gene or cell)
    for i in range(n_items):
        target_vec = slice_func(targets, i).toarray().ravel()
        pred_vec = slice_func(predictions, i).toarray().ravel()

        # Compute error metrics
        mae_vals[i] = np.mean(np.abs(target_vec - pred_vec))
        mse_vals[i] = np.mean(np.square(target_vec - pred_vec))

        # Compute correlation metrics with error handling
        try:
            pearson_corrs[i], _ = pearsonr(target_vec, pred_vec)
            spearman_corrs[i], _ = spearmanr(target_vec, pred_vec)
        except (ValueError, RuntimeWarning):
            # Handle cases where correlation cannot be computed (e.g., constant values)
            pearson_corrs[i], spearman_corrs[i] = np.nan, np.nan

    # Compile all metrics into a dictionary
    metrics = {
        label: item_names,
        "pearson_correlation": pearson_corrs,
        "spearman_correlation": spearman_corrs,
        "target_mean": target_means,
        "predicted_mean": pred_means,
        "mae": mae_vals,
        "mse": mse_vals,
        "rmse": np.sqrt(mse_vals),
        "target_sparsity": target_sparsity,
        "predicted_sparsity": pred_sparsity,
        "sparsity_difference": np.abs(target_sparsity - pred_sparsity),
    }

    return pd.DataFrame(metrics)


def calc_test_metrics(
    targets: sparse.csr_matrix,
    predictions: sparse.csr_matrix,
    gene_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate comprehensive evaluation metrics for gene expression predictions.

    This function computes three levels of metrics:
    1. Global metrics: Overall performance across all genes and cells
    2. Gene-wise metrics: Performance for each individual gene
    3. Cell-wise metrics: Performance for each individual cell

    :param targets: True expression values as sparse CSR matrix (cells x genes)
    :type targets: sparse.csr_matrix
    :param predictions: Predicted expression values as sparse CSR matrix (cells x genes)
    :type predictions: sparse.csr_matrix
    :param gene_names: Optional list of gene names. If None, generates default names
    :type gene_names: Optional[list[str]]

    :return: Tuple containing (global_metrics, gene_wise_metrics, cell_wise_metrics)
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    n_cells, n_genes = targets.shape

    # Generate default names if not provided
    cell_names = [f"cell_{i+1}" for i in range(n_cells)]
    if gene_names is None:
        gene_names = [f"gene_{i + 1}" for i in range(n_genes)]

    # Calculate global metrics on the entire dataset
    # Flatten sparse matrices to 1D arrays for global comparison
    targets_flat = targets.A1
    preds_flat = predictions.A1

    # Compute global correlation metrics
    pearson_corr, _ = pearsonr(targets_flat, preds_flat)
    spearman_corr, _ = spearmanr(targets_flat, preds_flat)

    # Compute global error metrics
    mae = mean_absolute_error(targets_flat, preds_flat)
    mse = mean_squared_error(targets_flat, preds_flat)
    rmse = np.sqrt(mse)

    # Compute global sparsity metrics
    target_sparsity = np.sum(targets_flat == 0) / len(targets_flat)
    pred_sparsity = np.sum(preds_flat == 0) / len(preds_flat)
    sparsity_diff = np.abs(target_sparsity - pred_sparsity)

    # Create global metrics DataFrame
    global_metrics = pd.DataFrame(
        {
            "pearson_correlation": [pearson_corr],
            "spearman_correlation": [spearman_corr],
            "mae": [mae],
            "mse": [mse],
            "rmse": [rmse],
            "target_sparsity": [target_sparsity],
            "predicted_sparsity": [pred_sparsity],
            "sparsity_difference": [sparsity_diff],
        }
    )

    # Calculate gene-wise metrics (performance for each gene across all cells)
    gene_wise_metrics = _calc_axis_wise_metrics(
        targets,
        predictions,
        gene_names,
        axis=C.GENE_AXIS,
        label=C.GENE_LABEL,
    )

    # Calculate cell-wise metrics (performance for each cell across all genes)
    cell_wise_metrics = _calc_axis_wise_metrics(
        targets,
        predictions,
        cell_names,
        axis=C.CELL_AXIS,
        label=C.CELL_LABEL,
    )

    return global_metrics, gene_wise_metrics, cell_wise_metrics


def _validate_colors(colors: dict[str, str] | None) -> bool:
    """Validate that all given colors are in valid hexadecimal format.

    This function checks if all color values in the provided dictionary are valid
    hexadecimal color codes. It supports 3, 6, and 8 character hex codes (with hash symbol).

    :param colors: Dictionary mapping color names to hex color strings, or None
    :type colors: dict[str, str] | None

    :return: True if all colors are valid hex codes or colors is None, False otherwise
    :rtype: bool
    """
    if colors is None:
        return False

    # Regular expression pattern for hex color validation:
    # 1. ^: Start of string
    # 2. #: Hash symbol
    # 3. [A-Fa-f0-9]{n}: Hexadecimal characters of length n (3, 6, or 8)
    # 4. $: End of string
    return all(re.match(C.HEX_COLOR_PATTERN, color) for color in colors.values())


def plot_value_over_period(
    data: pd.DataFrame,
    period_col: str,
    value_col: str = "value",
    title: str = "",
    x_lab: str = "Period",
    y_lab: str = "Value",
    color_by: str | None = None,
    custom_colors: dict[str, str] | None = None,
    legend_title: str = "",
    facet_by: str | None = None,
    save_to: str = "value_over_period.png",
) -> ggplot:
    """Create a line plot showing values over time periods with optional grouping and faceting.

    This function generates a line plot using plotnine (ggplot2 for Python) to visualize
    how values change over different time periods. It supports color grouping, custom colors,
    and faceting for more complex visualizations.

    :param data: DataFrame containing the data to plot
    :type data: pd.DataFrame
    :param period_col: Name of the column containing period/time information for x-axis
    :type period_col: str
    :param value_col: Name of the column containing values for y-axis
    :type value_col: str
    :param title: Title for the plot
    :type title: str
    :param x_lab: Label for the x-axis
    :type x_lab: str
    :param y_lab: Label for the y-axis
    :type y_lab: str
    :param color_by: Column name(s) to use for color grouping lines
    :type color_by: str | None
    :param custom_colors: Dictionary mapping group names to hex color codes
    :type custom_colors: dict[str, str] | None
    :param legend_title: Title for the color legend
    :type legend_title: str
    :param facet_by: Column name to use for creating separate plot panels
    :type facet_by: str | None

    :return: A ggplot object representing the configured plot
    :rtype: ggplot
    """
    plot = (
        ggplot(data, aes(x=period_col, y=value_col, color=color_by))
        + geom_point()
        + labs(title=title, x=x_lab, y=y_lab)
    )

    # Configure color scaling if color grouping is specified
    if color_by is not None:
        if _validate_colors(custom_colors):
            # Use custom colors if they are valid hex codes
            plot = plot + scale_color_manual(
                values=custom_colors,
                name=legend_title,
            )
        else:
            # Fall back to default color palette
            plot = plot + scale_color_brewer(
                type=C.DEFAULT_COLOR_TYPE,
                palette=C.DEFAULT_COLOR_PALETTE,
                name=legend_title,
            )

    plot = (
        plot
        + theme_minimal()
        + theme(
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
        )
    )

    if facet_by is not None and facet_by in data.columns:
        plot = plot + facet_wrap(f"~{facet_by}")

    ggsave(plot, save_to, dpi=C.PLOT_DPI)

    return plot


def plot_targets_vs_predictions(
    data: pd.DataFrame,
    title: str = "",
    x_lab: str = "Targets",
    y_lab: str = "Predictions",
    save_to: str = "targets_vs_predictions.png",
) -> ggplot:
    """Create a scatter plot of targets vs predictions with a diagonal reference line.

    This function generates a scatter plot comparing target values against predicted values,
    with a diagonal reference line indicating perfect predictions. The plot is saved to a file
    with customizable styling and labels.

    :param data: DataFrame containing "targets" and "predictions" columns
    :type data: pd.DataFrame
    :param title: Title for the plot
    :type title: str
    :param x_lab: Label for the x-axis
    :type x_lab: str
    :param y_lab: Label for the y-axis
    :type y_lab: str
    :param save_to: Filename/path where the plot will be saved
    :type save_to: str

    :return: A ggplot object representing the configured plot
    :rtype: ggplot

    :raises KeyError: If "targets" or "predictions" columns are missing from data
    :raises ValueError: If data is empty or contains invalid values
    """
    missing_columns = [col for col in C.REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: '{missing_columns}'.")

    if data.empty:
        raise ValueError("Input data cannot be empty.")

    plot = (
        ggplot(data, aes(x="targets", y="predictions"))
        + geom_point(fill=C.POINT_COLOR)
        + labs(title=title, x=x_lab, y=y_lab)
    )

    # Calculate bounds for diagonal reference line (perfect prediction line)
    min_val = min(data["targets"].min(), data["predictions"].min())
    max_val = max(data["targets"].max(), data["predictions"].max())
    diagonal = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})

    # Add diagonal reference line to the plot
    plot = plot + geom_line(
        data=diagonal,
        mapping=aes(x="x", y="y"),
        color=C.DIAGONAL_COLOR,
        linetype="dashed",
        alpha=C.DIAGONAL_ALPHA,
    )

    plot = (
        plot
        + theme_minimal()
        + theme(
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
        )
    )

    ggsave(plot, save_to, dpi=C.PLOT_DPI)

    return plot


def plot_frequency(
    data: pd.DataFrame,
    data_col: str,
    title: str = "",
    x_lab: str = "Value",
    y_lab: str = "Count",
    save_to: str = "value_frequency.png",
) -> ggplot:
    """Create a histogram plot showing the frequency distribution of a specified column.

    This function generates a histogram visualization using plotnine (ggplot2 for Python)
    with customizable styling and automatically saves the plot to a file. The plot uses
    a minimal theme with white background and configurable colors from constants.

    :param data: DataFrame containing the data to be plotted
    :type data: pd.DataFrame
    :param data_col: Name of the column in the DataFrame to create histogram for
    :type data_col: str
    :param title: Title to display at the top of the plot
    :type title: str, optional
    :param x_lab: Label for the x-axis of the plot
    :type x_lab: str, optional
    :param y_lab: Label for the y-axis of the plot
    :type y_lab: str, optional
    :param save_to: File path where the plot should be saved (including extension)
    :type save_to: str, optional

    :return: The generated ggplot object for further customization if needed
    :rtype: ggplot

    :raises KeyError: If data_col is not found in the DataFrame columns
    :raises ValueError: If the DataFrame is empty or data_col contains no valid data
    """
    if data_col not in data.columns:
        raise KeyError(f"Missing required column: '{data_col}'.")

    if data.empty:
        raise ValueError("Input data cannot be empty.")

    plot = (
        ggplot(data, aes(x=data_col))
        + geom_histogram(fill=C.HIST_FILL_COLOR, color=C.HIST_BORDER_COLOR)
        + labs(title=title, x=x_lab, y=y_lab)
    )

    plot = (
        plot
        + theme_minimal()
        + theme(
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
        )
    )

    ggsave(plot, save_to, dpi=C.PLOT_DPI)

    return plot
