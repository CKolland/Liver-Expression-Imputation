import argparse

import test_model._constants as C
from test_model.test_model import test_model


def main():
    """_summary_"""
    # --------------------
    #     Setup parser
    # --------------------
    parser = argparse.ArgumentParser(description=C.PARSER_DESC)

    parser.add_argument(
        C.DATA_LONG,
        C.DATA_SHORT,
        type=str,
        help=C.DATA_HELP,
    )
    parser.add_argument(
        C.LAYER_LONG,
        C.LAYER_SHORT,
        default=C.LAYER_DEFAULT,
        help=C.LAYER_HELP,
    )
    parser.add_argument(
        C.MASK_LONG,
        C.MASK_SHORT,
        type=str,
        help=C.MASK_HELP,
    )
    parser.add_argument(
        C.MODEL_LONG,
        C.MODEL_SHORT,
        type=str,
        help=C.MODEL_HELP,
    )
    parser.add_argument(
        C.OUTPUT_LONG,
        C.OUTPUT_SHORT,
        type=str,
        help=C.OUTPUT_HELP,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    test_model(
        args.test_data,
        args.imputation_mask,
        args.model,
        args.output,
        args.layer,
    )


if __name__ == "__main__":
    main()
