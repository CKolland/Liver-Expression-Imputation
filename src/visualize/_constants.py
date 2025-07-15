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
#     `concat` command
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
