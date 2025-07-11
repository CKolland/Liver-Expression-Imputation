import argparse
import os

# Enable expandable CUDA memory segments to reduce fragmentation
# Helps prevent CUDA OOM errors in multi-phase training (e.g., K-Fold CV)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import fit_model._constants as C
from fit_model.fit_model import fit_model, test_model_fitting


def main():
    """_summary_"""
    # --------------------
    #     Setup parser
    # --------------------
    parser = argparse.ArgumentParser(description=C.PARSER_DESC)

    parser.add_argument(
        C.CONFIG_LONG,
        C.CONFIG_SHORT,
        type=str,
        help=C.CONFIG_HELP,
    )
    parser.add_argument(
        C.OUTPUT_LONG,
        C.OUTPUT_SHORT,
        type=str,
        help=C.OUTPUT_HELP,
    )
    parser.add_argument(
        C.TEST_LONG,
        C.TEST_SHORT,
        action="store_true",
        help=C.TEST_HELP,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    if args.test:
        test_model_fitting()
    else:
        if not args.config or not args.output:
            parser.error(C.PARSER_ERR)
        fit_model(args.config, args.output)


if __name__ == "__main__":
    main()
