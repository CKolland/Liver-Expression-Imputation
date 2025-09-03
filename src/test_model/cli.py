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
        C.CONFIG_LONG,
        C.CONFIG_SHORT,
        type=str,
        help=C.CONFIG_HELP,
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

    test_model(args.config, args.model, args.output)


if __name__ == "__main__":
    main()
