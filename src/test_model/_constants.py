# --------------------
#     Parser setup
# --------------------

PARSER_DESC = "Functionality to test a trained model and save results to a structured file format."

# Config argument
CONFIG_LONG = "--config"
CONFIG_SHORT = "-c"
CONFIG_HELP = "Path to the config file."
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
