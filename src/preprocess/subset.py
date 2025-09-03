import logging

from anndata import read_h5ad
import numpy as np

import preprocess._constants as C
from utils.io import assert_path


def subset_adata(path_to_adata: str, pct_rows: float, pct_cols: float, shuffle: bool):
    """_summary_"""
    # Setup custom logging
    logging.basicConfig(
        level=getattr(logging, C.LOGGING_LEVEL),
        format=C.LOGGING_FORMAT,
        datefmt=C.LOGGING_DATEFMT,
    )

    # Validate inputs
    if not isinstance(path_to_adata, str):
        raise TypeError(C.ADATA_PATH_ERR)

    if not (0.0 <= pct_rows <= 1.0):
        raise ValueError(C.PERCENTAGE_ROWS_ERR)

    if not (0.0 <= pct_cols <= 1.0):
        raise ValueError(C.PERCENTAGE_COLS_ERR)

    # Load the AnnData file
    adata_file = assert_path(path_to_adata, assert_dir=False)
    adata = read_h5ad(adata_file)
    logging.info(C.SUBSET_READ_MSG.format(adata=adata))

    # Calculate number of rows and columns to keep
    n_rows = adata.n_obs
    n_cols = adata.n_vars

    n_rows_keep = int(n_rows * pct_rows)
    logging.info(C.SUBSET_ROWS_USED_MSG.format(n_rows=n_rows_keep))

    n_cols_keep = int(n_cols * pct_cols)
    logging.info(C.SUBSET_COLS_USED_MSG.format(n_cols=n_cols_keep))

    # Ensure we keep at least 1 row and 1 column if percentages are > 0
    if pct_rows > 0 and n_rows_keep == 0:
        n_rows_keep = 1
    if pct_cols > 0 and n_cols_keep == 0:
        n_cols_keep = 1

    # Select row indices
    if shuffle:
        row_indices = np.random.choice(n_rows, size=n_rows_keep, replace=False)
        row_indices = np.sort(row_indices)  # Sort to maintain some order
    else:
        row_indices = np.arange(n_rows_keep)

    # Select column indices
    if shuffle:
        col_indices = np.random.choice(n_cols, size=n_cols_keep, replace=False)
        col_indices = np.sort(col_indices)  # Sort to maintain some order
    else:
        col_indices = np.arange(n_cols_keep)

    # Subset the AnnData object
    adata_subset = adata[row_indices, col_indices].copy()

    save_path = adata_file.with_stem(adata_file.stem + "_subset")
    adata_subset.write(save_path, compression="gzip")
    logging.info(C.SUBSET_SAVE_DATA_MSG.format(save_path=save_path))
