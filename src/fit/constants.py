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

# Torch
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
TORCH_OPTIM = {"adam": optim.Adam}
