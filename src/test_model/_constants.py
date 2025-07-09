# --------------------
#     Parser setup
# --------------------

PARSER_DESC = "Functionality to test a trained model and save results to a structured file format."

# Config argument
DATA_LONG = "--test-data"
DATA_SHORT = "-t"
DATA_HELP = "Path to the testset file."
LAYER_LONG = "--layer"
LAYER_SHORT = "-l"
LAYER_DEFAULT = "log1p"
LAYER_HELP = "Name of the layer that will be used for the run."
MASK_LONG = "--imputation-mask"
MASK_SHORT = "--i"
MASK_HELP = "Path to imputation mask file (.csv)."
MODEL_LONG = "--model"
MODEL_SHORT = "-m"
MODEL_HELP = "Path to the saved model (.pth)."
OUTPUT_LONG = "--output"
OUTPUT_SHORT = "-o"
OUTPUT_HELP = "Path to the output directory."

# ---------------------
#     Logging setup
# ---------------------

LOGGING_LVL_CONSOLE = "INFO"
LOGGING_LVL_FILE = "DEBUG"
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s |> %(message)s"
LOGGING_DATEFMT = "%Y-%m-%d %H:%M:%S"
