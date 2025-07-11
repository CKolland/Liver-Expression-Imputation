from datetime import datetime

import anndata as ad
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import confy
from utils.io import assert_path, ImputationDataset, load_toml, setup_logging
from utils.fit import TestingPipeline
from utils.model import MLP


def test_model(
    setup_path: str,
    model_path: str,
    out_path: str,
):
    """_summary_

    :param str config_path: _description_
    :param str out_path: _description_
    """
    # Verify paths
    setup_file = assert_path(setup_path, assert_dir=False)  # File
    model_file = assert_path(model_path, assert_dir=False)
    out_dir = assert_path(out_path)  # Directory

    # Load model config here to get model name
    run_setup = load_toml(setup_file)
    model_config = confy.setup_model(run_setup["model"])

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

    # Load test data
    testsets = []
    test_config = confy.setup_dataset(run_setup["test_data"])
    for testset in test_config:
        test_data_config = test_config.config[testset]
        test_data = ad.read_h5ad(test_data_config["path"])
        logger.info(f"📁 Training data loaded successfully: '{test_data_config['path']}'.")

        # Choose data layer if provided
        if test_data_config["layer"] is not None:
            test_data.X = test_data.layers[test_data_config["layer"]]
            logger.info(f"Selected {test_data_config["layer"]} as data layer.")

        logger.debug(test_data)
        testsets.append(test_data)

    # Load imputation mask
    masking_config = confy.setup_dataset(run_setup["imputation_mask"])
    imputation_mask = pd.read_csv(masking_config.path, header=masking_config.header)
    imputation_mask = imputation_mask[0].tolist()
    logger.info(f"📁 Mask for imputation loaded successfully: '{masking_config.path}'.")

    test_datasets = []
    for testset in testsets:
        test_dataset = ImputationDataset(testset, imputation_mask)
        logger.info("Created testing dataset.")
        logger.debug(test_dataset)
        test_datasets.append(test_dataset)

    logger.info("----")

    # -------------------
    #     Setup model
    # -------------------

    loaded_model = torch.load(model_file, map_location=device)
    # If state dict is loaded
    if isinstance(loaded_model, dict):
        mlp = MLP(model_config.to_torch(), device)
        logger.info("Built model successfully.")
        logger.debug(mlp)

        mlp.load_state_dict(loaded_model)
    # If complete model is loaded
    else:
        mlp = loaded_model

    logger.info("----")

    # ------------------
    #     Test model
    # ------------------

    # Load training config
    train_config = confy.setup_training(run_setup["training"])

    for test_dataset in test_datasets:
        # Create test data loader
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        pipeline = TestingPipeline(test_loader, mlp, device, logger)
        results = pipeline.test()
        adata = pipeline.create_anndata(results)

        # Save AnnData object
        adata.write(run_dir / f"{model_config.name}_test_results_{now}.h5ad")
