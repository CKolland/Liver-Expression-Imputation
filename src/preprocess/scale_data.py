import os.path as path
import logging
from pathlib import Path
from typing import Optional

from anndata import AnnData, read_h5ad
import numpy as np
from scipy.sparse import csr_matrix

import preprocess.constants as C
from utils.io_utils import verify_path


def prep_data(adata: AnnData, data_layer: Optional[str]) -> csr_matrix:
    """Prepares and returns a sparse matrix from an AnnData object.

    If `data_layer` is None, the main matrix (`adata.X`) is returned.
    If a valid `data_layer` is specified and exists in `adata.layers`, that layer is returned.
    Otherwise, an error is logged.

    :param AnnData adata: The AnnData object containing gene expression data.
    :param str or None data_layer: The name of the layer in `adata.layers` to use. If None, uses `adata.X`.

    :return: The selected data matrix as a sparse CSR matrix.
    :rtype: scipy.sparse.csr_matrix
    """
    if data_layer is None:
        data = adata.X
    elif data_layer in adata.layers:
        data = adata.layers[data_layer]
    else:
        logging.error(C.SCALE_DLAYER_ERR.format(data_layer=data_layer))

    return data


def scale_log1p(adata: AnnData, data_layer: Optional[str]) -> AnnData:
    """Applies a log(1 + x) transformation to a specified data layer in an AnnData object.

    If a 'log1p' layer already exists in the object, no transformation is applied.
    Otherwise, the specified layer is transformed and stored as a new 'log1p' layer.

    :param AnnData adata: The AnnData object containing the input data.
    :param str or None data_layer: Name of the data layer in `adata.layers` to transform.

    :return: The updated AnnData object with the new 'log1p' layer (if added).
    :rtype: AnnData
    """
    # Only add log1p layer if it doesn't already exist
    if "log1p" not in adata.layers:
        # Data layer must exist to scale data
        data = prep_data(adata, data_layer=data_layer)
        adata.layers["log1p"] = np.log1p(data)
        logging.info(C.SCALE_SUCCESS_MSG.format(mode="log1p", adata=adata))

    else:
        # If already present, do nothing
        logging.warning(C.SCALE_LAYER_WARN)

    return adata


def scale_data(adata_path: str, scale_mode: str, data_layer: str):
    """Loads an AnnData file, applies a specified scaling transformation, and saves the result.

    Currently supports only the 'log1p' transformation mode. The transformed data
    is saved to a new file named after the original input.

    :param str adata_path: Path to the input AnnData (.h5ad) file.
    :param str scale_mode: Scaling mode to apply. Supported values: "log1p".
    :param str data_layer: Name of the data layer within the AnnData object to transform.

    :raises FileNotFoundError: If the input file does not exist.
    :raises ValueError: If the provided scale mode is unsupported.
    """
    # Setup custom logging
    logging.basicConfig(
        level=getattr(logging, C.LOGGING_LEVEL),
        format=C.LOGGING_FORMAT,
        datefmt=C.LOGGING_DATEFMT,
    )

    # Load the AnnData file
    verify_path(adata_path)
    adata = read_h5ad(adata_path)
    logging.info(C.SCALE_READ_MSG)

    if scale_mode == C.SCALE_MODE_CHOICES[0]:  # Mode: log1p
        adata = scale_log1p(adata, data_layer)

        # Save with new filename indicating log1p transformation
        adata.write(adata_path)
        logging.info(C.SCALE_SAVE_MSG.format(mode=scale_mode, out_path=adata_path))

    else:
        logging.error(C.SCALE_MODE_ERR.format(mode=scale_mode))
