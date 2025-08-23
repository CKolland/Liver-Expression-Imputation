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
#     `baseline` command
# ------------------------

# Command parser setup
SUB_COMMAND_BASELINE = "baseline"
BASELINE_HELP = (
    "Calculate gene-wise test metrics for the baseline against the predicted values."
)

TEST_BASELINE_LONG = "--adata"
TEST_BASELINE_SHORT = "-d"
TEST_BASELINE_HELP = "Paths to AnnData (.h5ad) file."
TEST_BASELINE_LONG = "--masks"
TEST_BASELINE_SHORT = "-m"
TEST_BASELINE_DEFAULT = None
TEST_BASELINE_HELP = "Path to data frame that contains the masks."

# ------------------------
#     `threshold` command
# ------------------------

# Command parser setup
SUB_COMMAND_THRESHOLD = "threshold"
THRESHOLD_HELP = (
    "Calculate gene-wise test metrics with applied threshold to predicted values."
)

THRESHOLD_ADATA_LONG = "--adata"
THRESHOLD_ADATA_SHORT = "-d"
THRESHOLD_ADATA_HELP = "Paths to AnnData (.h5ad) file."
THRESHOLD_MASKS_LONG = "--masks"
THRESHOLD_MASKS_SHORT = "-m"
THRESHOLD_MASKS_DEFAULT = None
THRESHOLD_MASKS_HELP = "Path to data frame that contains the masks."
THRESHOLD_LONG = "--threshold"
THRESHOLD_SHORT = "-t"
THRESHOLD_DEFAULT = 0.05
THRESHOLD_HELP = "Threshold that is applied to predicted values. Defaults to 0.05."
