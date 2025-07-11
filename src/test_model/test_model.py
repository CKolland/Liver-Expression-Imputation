from datetime import datetime

import anndata as ad
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.io import assert_path, ImputationDataset, setup_logging
from utils.fit import TestingPipeline


def test_model(
    testset_path: str,
    mask_path: str,
    model_path: str,
    out_path: str,
    layer: str | None = None,
):
    """_summary_

    :param str config_path: _description_
    :param str out_path: _description_
    """
    # Verify paths
    testset_file = assert_path(testset_path, assert_dir=False)  # File
    mask_file = assert_path(mask_path, assert_dir=False)
    model_file = assert_path(model_path, assert_dir=False)
    out_dir = assert_path(out_path)  # Directory

    # Create run directory
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = out_dir / f"model_test_run_{now}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------
    #     Setup logging
    # ---------------------

    logger = setup_logging(run_dir / f"model_test_{now}.log", "test_model")
    logger.info("✅ Setup complete.")

    # -----------------
    #     Load data
    # -----------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Supported device for this run: {device}.")
    logger.info("----")

    # Load training data
    test_data = ad.read_h5ad(testset_file)
    logger.info(f"📁 Test data loaded successfully: '{testset_file}'.")

    # Choose data layer if provided
    if layer is not None:
        test_data.X = test_data.layers[layer]
        logger.info(f"Selected {layer} as data layer.")

    logger.debug(test_data)

    # Load imputation mask
    imputation_mask = pd.read_csv(mask_file, header=None)
    imputation_mask = imputation_mask[0].tolist()
    logger.info(f"📁 Mask for imputation loaded successfully: '{mask_file}'.")

    test_dataset = ImputationDataset(test_data, imputation_mask)
    logger.info("Created testing dataset.")
    logger.debug(test_dataset)
    logger.info("----")

    # -------------------
    #     Setup model
    # -------------------

    mlp = torch.load(model_file, map_location=device)
    logger.info("Model successfully loaded.")
    logger.debug(mlp)
    logger.info("----")

    # ------------------
    #     Test model
    # ------------------

    # Create test data loader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    pipeline = TestingPipeline(test_loader, mlp, device, logger)
    results = pipeline.test()
    adata = pipeline.create_anndata(results)

    # Save AnnData object
    adata.write(run_dir / f"test_results_{now}.h5ad")
