# Parent parser setup
PARSER_PREP_DESC = "Functionality to preprocess '.h5ad' data for expression imputation."
SUBPARSERS_PREP_DEST = "command"

# Logging setup
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "[%(levelname)s] %(asctime)s |> %(message)s"
LOGGING_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ------------------------
#     `concat` command
# ------------------------

# Command parser setup
SUB_COMMAND_CONCAT = "concat"
CONCAT_HELP = (
    "Concatenate multiple AnnData objects (adds unique sample identifier if necessary)."
)
CONCAT_ADATAS_LONG = "--adatas"
CONCAT_ADATAS_SHORT = "-d"
CONCAT_ADATAS_HELP = "List of paths to AnnData (.h5ad) files."
CONCAT_OUTPUT_LONG = "--output"
CONCAT_OUTPUT_SHORT = "-o"
CONCAT_OUTPUT_DEFAULT = "./adata_concat.h5ad"
CONCAT_OUTPUT_HELP = "Path to save the AnnData object. Default: 'adata_concat.h5ad'."

# Process constants
CONCAT_INDEX_UNIQUE = "_"
CONCAT_SUCCESS_MSG = "Concatenated all AnnData objects:\n{adata_concat}."
CONCAT_DUPLICATE_WARNING = "Identical `obs` entries where found, added unique index."
CONCAT_WRITE_MSG = "Wrote concatenated AnnData object to '{out_path}'."
CONCAT_READ_MSG = "Read all AnnData objects."

# -----------------------
#     `scale` command
# -----------------------

# Command parser setup
SUB_COMMAND_SCALE = "scale"
SCALE_HELP = (
    "Scale expression data from an AnnData object. Adds scaled data to new layer."
)
SCALE_ADATA_LONG = "--adata"
SCALE_ADATA_SHORT = "-d"
SCALE_ADATA_HELP = "Path to AnnData (.h5ad) file."
SCALE_MODE_LONG = "--mode"
SCALE_MODE_SHORT = "-m"
SCALE_MODE_CHOICES = ["log1p"]
SCALE_MODE_DEFAULT = "log1p"
SCALE_MODE_HELP = f"Data scaling options."
SCALE_LAYER_LONG = "--layer"
SCALE_LAYER_SHORT = "-l"
SCALE_LAYER_DEFAULT = None
SCALE_LAYER_HELP = "Data layer that is used for data scaling. Default: 'raw'"

# Process constants
SCALE_READ_MSG = "Read AnnData object."
SCALE_DLAYER_ERR = (
    "Failed to scale data: Data layer must exist.\n{data_layer} is not a layer!"
)
SCALE_SUCCESS_MSG = "Data scaled successfully. Added to layer {mode}.\n{adata}"
SCALE_LAYER_WARN = "'log1p' layer already present, no changes made."
SCALE_MODE_ERR = "Failed to scale data: '{mode}' is invalid mode."
SCALE_SAVE_MSG = "Added {mode} layer and saved to '{out_path}'."

# -------------------------
#     `testset` command
# -------------------------

# Command parser setup
SUB_COMMAND_TESTSET = "testset"
TESTSET_HELP = "Create train/test split from a given AnnData object."
TESTSET_ADATA_LONG = "--adata"
TESTSET_ADATA_SHORT = "-d"
TESTSET_ADATA_HELP = "Path to AnnData (.h5ad) file."
TESTSET_SPLIT_LONG = "--split"
TESTSET_SPLIT_SHORT = "-s"
TESTSET_SPLIT_DEFAULT = 0.1
TESTSET_SPLIT_HELP = "Percentage used to split the data. Default: 0.1 (10%%)"

# Process constants
TESTSET_READ_MSG = "Read AnnData object.\n{adata}"
TESTSET_CREATED_SPLIT_MSG = "Created random split of size: {size}"
TESTSET_TESTSET_MSG = "Created testing dataset.\n{test_data}"
TESTSET_TRAINSET_MSG = "Created training dataset.\n{train_data}"
TESTSET_SAVE_MSG = "Saved train/test split to '{out_dir}'."

# ------------------------
#     `subset` command
# ------------------------

# Command parser setup
SUB_COMMAND_SUBSET = "subset"
SUBSET_HELP = "Create subset from a given AnnData object."
SUBSET_ADATA_LONG = "--adata"
SUBSET_ADATA_SHORT = "-d"
SUBSET_ADATA_HELP = "Path to AnnData (.h5ad) file."
SUBSET_ROWS_LONG = "--rows"
SUBSET_ROWS_SHORT = "-r"
SUBSET_ROWS_DEFAULT = 0.1
SUBSET_ROWS_HELP = "Percentage of rows used in the subset."
SUBSET_COLS_LONG = "--columns"
SUBSET_COLS_SHORT = "-c"
SUBSET_COLS_DEFAULT = 0.1
SUBSET_COLS_HELP = "Percentage of columns used in the subset."
SUBSET_SHUFFLE_LONG = "--shuffle"
SUBSET_SHUFFLE_SHORT = "-s"
SUBSET_SHUFFLE_HELP = "Whether the used rows and columns are taken randomly."

# Process constants
ADATA_PATH_ERR = "Must provide path to '.h5ad' file."
PERCENTAGE_ROWS_ERR = "Percante of rows must be between 0.0 and 1.0."
PERCENTAGE_COLS_ERR = "Percentage of cols must be between 0.0 and 1.0."
SUBSET_READ_MSG = "Read AnnData object.\n{adata}"
SUBSET_ROWS_USED_MSG = "Use {n_rows} rows to create the subset."
SUBSET_COLS_USED_MSG = "Use {n_cols} cols to create the subset."
SUBSET_SAVE_DATA_MSG = "Successfully wrote subset file to '{save_path}'."
