# ---------------------------
#     Parent parser setup
# ---------------------------

PARSER_VIS_DESC = "Functionality to visualize results from this DL pipeline."
SUBPARSERS_VIS_DEST = "command"

# ---------------------
#     Logging setup
# ---------------------

LOGGING_LVL_CONSOLE = "INFO"
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s |> %(message)s"
LOGGING_DATEFMT = "%Y-%m-%d %H:%M:%S"


# ------------------------
#     `test` command
# ------------------------

# Command parser setup
SUB_COMMAND_TEST = "test"
TEST_HELP = "Visualize the test results."

TEST_ADATA_LONG = "--adata"
TEST_ADATA_SHORT = "-d"
TEST_ADATA_HELP = "Paths to AnnData (.h5ad) file."
TEST_MASKS_LONG = "--masks"
TEST_MASKS_SHORT = "-m"
TEST_MASKS_DEFAULT = None
TEST_MASKS_HELP = "Path to data frame that contains the masks."

# ------------------------
#     `threshold` command
# ------------------------

# Command parser setup
SUB_COMMAND_TRESHOLD = "treshold"
TRESHOLD_HELP = (
    "Calculate gene-wise test metrics with applied threshold to predicted values."
)

TRESHOLD_ADATA_LONG = "--adata"
TRESHOLD_ADATA_SHORT = "-d"
TRESHOLD_ADATA_HELP = "Paths to AnnData (.h5ad) file."
TRESHOLD_MASKS_LONG = "--masks"
TRESHOLD_MASKS_SHORT = "-m"
TRESHOLD_MASKS_DEFAULT = None
TRESHOLD_MASKS_HELP = "Path to data frame that contains the masks."
TRESHOLD_MASKS_LONG = "--threshold"
TRESHOLD_MASKS_SHORT = "-t"
TRESHOLD_MASKS_DEFAULT = 0.05
TRESHOLD_MASKS_HELP = "Threshold that is applied to predicted values. Defaults to 0.05."
