import torch.nn as nn
import torch.optim as optim

# --------------------
#     Parser setup
# --------------------

PARSER_DESC = "Functionality to train the model for expression imputation based on given configuraiton."

# Config argument
CONFIG_LONG = "--config"
CONFIG_SHORT = "-c"
CONFIG_HELP = "Path to the config file."
OUTPUT_LONG = "--output"
OUTPUT_SHORT = "-o"
OUTPUT_HELP = "Path to the output directory."
TEST_LONG = "--test"
TEST_SHORT = "-t"
TEST_HELP = "Performs TEST run with synthetic data."

PARSER_ERR = "The following arguments are required when not running test mode: --config, --output"

# ---------------------
#     Logging setup
# ---------------------

LOGGING_LVL_CONSOLE = "INFO"
LOGGING_LVL_FILE = "DEBUG"
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s |> %(message)s"
LOGGING_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ---------------------
#     Torch mapping
# ---------------------

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
