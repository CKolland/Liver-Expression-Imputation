# Parent parser setup
PARSER_PREP_DESC = "Functionality to preprocess '.h5ad' data for expression imputation."
SUBPARSERS_PREP_DEST = "command"

# Logging setup
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s || %(levelname)s |> %(message)s"
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

# ------------------------
#     `testset` command
# ------------------------

# Command parser setup
SUB_COMMAND_SUBSET = "testset"
SUBSET_HELP = "Create train/test split from a given AnnData object."
SUBSET_ADATA_LONG = "--adata"
SUBSET_ADATA_SHORT = "-d"
SUBSET_ADATA_HELP = "Path to AnnData (.h5ad) file."
SUBSET_SPLIT_LONG = "--split"
SUBSET_SPLIT_SHORT = "-s"
SUBSET_SPLIT_HELP = "Percentage used to split the data. Default: 0.1 (10%%)"

# Process constants
SUBSET_READ_MSG = "Read AnnData object.\n{adata}"
SUBSET_CREATED_SPLIT_MSG = "Created random split of size: {size}"
SUBSET_TESTSET_MSG = "Created testing dataset.\n{test_data}"
SUBSET_TRAINSET_MSG = "Created training dataset.\n{train_data}"
SUBSET_SAVE_MSG = "Saved train/test split to '{out_dir}'."
