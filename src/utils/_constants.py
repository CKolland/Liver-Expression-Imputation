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

DATA_NOT_DF_ERR = "Data must be pandas DataFrame."
PERIOD_COL_NOT_FOUND_ERR = "Period column '{col}' not found in DataFrame."
VALUE_COLS_NOT_FOUND_ERR = "Not all value columns found in DataFrame: {missing}"
FACET_COL_NOT_FOUND_ERR = "Facet column '{col}' not foind in DataFrame."
ALLOWED_GEOMS = ["line", "point", "both"]
UNKNOWN_GEOM_ERR = f"Plot type must be one of: {ALLOWED_GEOMS}"
UNKNOWN_CONFIG_ARG_ERR = "Unknown configuration argument: '{key}'"
