import argparse

import fit.constants as C
from fit.fit_model import fit_model


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
        required=True,
        help=C.CONFIG_HELP,
    )
    parser.add_argument(
        C.OUTPUT_LONG,
        C.OUTPUT_SHORT,
        type=str,
        required=True,
        help=C.OUTPUT_HELP,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    fit_model(args.config, args.output)


if __name__ == "__main__":
    main()
