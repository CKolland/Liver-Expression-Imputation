import logging
import os
import sys

import anndata as ad
import numpy as np


def subset_adata(adata_path: str, n_rows: int):
    """Subsets an AnnData object by randomly selecting a specified number of rows (observations)
    and saves the subset to a new file.

    :param str adata_path: Path to the input AnnData (.h5ad) file.
    :param int n_rows: Number of rows (observations) to randomly select for the subset.
    """

    # Read AnnData object
    adata = ad.read_h5ad(adata_path)

    if adata.n_obs >= n_rows:  # Make sure enough rows are present
        random_indices = np.random.choice(adata.n_obs, size=n_rows, replace=False)

        # Subset the AnnData object using the selected indices
        adata_subset = adata[random_indices, :].copy()
        adata_subset.write("adata_subset.h5ad")


def setup_logging(path_to_log: str) -> logging.Logger:
    """Set up and configure logging to both console and file.

    :param str path_to_log: Path to the log file where logs will be written

    :return: Configured logger instance
    :rtype: logging.Logger
    """

    logger = logging.getLogger("data_integration")
    logger.setLevel(logging.DEBUG)  # Lowest log level (logs everything)

    # Custom formatter
    formatter = logging.Formatter(
        "%(asctime)s || LEVEL: %(levelname)s |> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Set handler specific log level
    console_handler.setFormatter(formatter)  # Add custom formatter to handler
    logger.addHandler(console_handler)  # Add handler to the logger

    # Add file handler
    file_handler = logging.FileHandler(path_to_log)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def verify_path(path: str, is_dir: bool = False) -> str:
    """Verify that the given path exists and is of the expected type (file or directory).

    :param str path: The path to verify
    :param bool is_dir: (optional) If True, checks that the path is a directory. If False, checks
        that the path is a file

    :return: The verified path if it exists and matches the expected type
    :rtype: str

    :raises FileNotFoundError: If the path does not exist or does not match the expected type
    """

    if is_dir:
        if not os.path.isdir(path):  # Verify if path truly leads to directory
            err_msg = f"""
            Path must be a directory!\n
            `{path}` leads to a file.
            """
            raise FileNotFoundError(err_msg)
        else:
            return path
    else:
        if not os.path.isfile(path):  # Verify if path truly leads to file
            err_msg = f"""
            Path must lead to file!\n 
            `{path}` leads to a directory.
            """

            raise FileNotFoundError(err_msg)
        else:
            return path
