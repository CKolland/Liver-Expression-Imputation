import torch.nn as nn
import torch.optim as optim

# ------------------
#     `utils.io`
# ------------------

LOGGING_LVL_CONSOLE = "INFO"
LOGGING_LVL_FILE = "DEBUG"
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s |> %(message)s"
LOGGING_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ---------------------
#     `utils.confy`
# ---------------------

# Torch lookup tables
TORCH_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}
TORCH_LAYERS = {
    "linear": nn.Linear,
    "dropout": nn.Dropout,
}
TORCH_LOSS = {"mse": nn.MSELoss}
TORCH_OPTIM = {
    "adam": optim.Adam,
    "adamW": optim.AdamW,
}

# -------------------
#     `utils.vis`
# -------------------

PLOT_DPI = 300

# _calc_axis_wise_metrics
GENE_AXIS = 0
CELL_AXIS = 1

# calc_test_metrics
GENE_LABEL = "gene"
CELL_LABEL = "cell"

# _validate_colors
HEX_COLOR_PATTERN = r"^#([A-Fa-f0-9]{3}|[A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$"

# plot_value_over_period
DEFAULT_COLOR_PALETTE = "Paired"
DEFAULT_COLOR_TYPE = "qual"

# plot_targets_vs_predictions
REQUIRED_COLUMNS = ["targets", "predictions"]
POINT_COLOR = "#1D2951"
DIAGONAL_COLOR = "#FFD166"
DIAGONAL_ALPHA = 0.7

# plot_frequency
HIST_FILL_COLOR = "#1D2951"
HIST_BORDER_COLOR = "#FFD166"
