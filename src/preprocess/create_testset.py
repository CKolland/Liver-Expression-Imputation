import logging
from pathlib import Path

from anndata import read_h5ad
import numpy as np

import preprocess._constants as C
from utils.io import assert_path


def create_testset(path_to_adata: str, split: float):
    """Splits a given AnnData object into a train/test set and saves them to disk.

    This function reads an AnnData object from a `.h5ad` file, randomly selects
    a subset of observations to form a test set based on the specified split ratio,
    and saves both the resulting train and test sets as separate `.h5ad` files.

    :param str path_to_adata: Path to the input `.h5ad` file containing the full dataset.
    :param float split: Proportion of observations to include in the test set (e.g., 0.2 for 20%).

    :raises FileNotFoundError: If the input file does not exist.
    :raises ValueError: If `split` is not in the range (0, 1).
    """
    # Setup custom logging
    logging.basicConfig(
        level=getattr(logging, C.LOGGING_LEVEL),
        format=C.LOGGING_FORMAT,
        datefmt=C.LOGGING_DATEFMT,
    )

    # Load the AnnData file
    adata_path = assert_path(path_to_adata)
    adata = read_h5ad(adata_path)
    logging.info(C.SUBSET_READ_MSG.format(adata=adata))

    # Create random test split
    split_size = adata.shape[0] * split
    random_samples = np.random.choice(adata.n_obs, size=int(split_size), replace=False)
    logging.info(C.SUBSET_CREATED_SPLIT_MSG.format(size=int(split_size)))

    # Create bool mask to identify samples used for testset
    test_mask = np.full(adata.n_obs, False, dtype=bool)
    test_mask[random_samples] = True

    # Create test dataset
    adata_test = adata[test_mask, :].copy()
    logging.info(C.SUBSET_TESTSET_MSG.format(test_data=adata_test))

    # Keep rest of training data
    adata_train = adata[~test_mask, :].copy()
    logging.info(C.SUBSET_TRAINSET_MSG.format(train_data=adata_train))

    adata_path = Path(adata_path)
    adata_test.write(f"{adata_path.parent}/{adata_path.stem}_test.h5ad")
    adata_train.write(f"{adata_path.parent}/{adata_path.stem}_train.h5ad")
    logging.info(C.SUBSET_SAVE_MSG.format(out_dir=adata_path.parent))
