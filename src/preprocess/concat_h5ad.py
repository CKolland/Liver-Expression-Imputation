import logging
from pathlib import Path

import anndata as ad

from preprocess import _constants as C
from utils.io import assert_path


def duplicated_obs_names(adata_objs: list[ad.AnnData]) -> bool:
    """Check if there are duplicated observation names across a list of AnnData objects.

    :param list[ad.AnnData] adata_objs: List of AnnData objects.

    :return: True if any duplicate observation names are found, False otherwise.
    :rtype: bool
    """
    # Indicates whether there are duplicate obs names
    duplicated = False

    seen = set()
    for adata in adata_objs:
        current = set(adata.obs_names)
        overlap = seen & current
        if overlap:
            duplicated = True
            break  # No need to further continue
        seen.update(current)

    return duplicated


def concat_h5ad(adata_paths: list[str], out_path: Path):
    """Concatenate multiple AnnData (.h5ad) files into a single AnnData object and write to disk.
    Handles duplicate observation names by making them unique if needed.

    :param list[str] adata_objs: List of paths to input AnnData files.
    :param Path out_path: Path to write the concatenated AnnData object.
    """
    # Setup custom logging
    logging.basicConfig(
        level=getattr(logging, C.LOGGING_LEVEL),
        format=C.LOGGING_FORMAT,
        datefmt=C.LOGGING_DATEFMT,
    )

    # Validate that all object paths are correct
    adata_objs = []
    for path in adata_paths:
        adata_objs.append(assert_path(path))

    # Load all input AnnData objects
    adatas = [ad.read_h5ad(p) for p in adata_objs]
    logging.info(C.CONCAT_READ_MSG)

    # Check for duplicated row indices
    if not duplicated_obs_names(adatas):
        # Concatenate AnnData objects
        adata_concat = ad.concat(adatas)
        logging.info(C.CONCAT_SUCCESS_MSG.format(adata_concat=adata_concat))
    else:
        # `index_unique` adds unique index depending on the dataset
        adata_concat = ad.concat(adatas, index_unique="_")
        logging.info(C.CONCAT_SUCCESS_MSG.format(adata_concat=adata_concat))
        logging.warning(C.CONCAT_DUPLICATE_WARNING)

    adata_concat.write(out_path)
    logging.info(C.CONCAT_WRITE_MSG.format(out_path=out_path))
