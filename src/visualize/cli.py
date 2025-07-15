import argparse

import visualize._constants as C
from visualize.visualize import visualize_test


def main():
    """_summary_"""
    # --------------------
    #     Setup parser
    # --------------------
    parser = argparse.ArgumentParser(description=C.PARSER_VIS_DESC)

    # Create a subparser object for multiple CLI commands
    subparsers = parser.add_subparsers(
        dest=C.SUBPARSERS_VIS_DEST,
        required=True,
    )

    # ----------------------
    #     `test` command
    # ----------------------

    # Visualizes test results
    parser_test = subparsers.add_parser(C.SUB_COMMAND_TEST, help=C.TEST_HELP)

    parser_test.add_argument(
        C.TEST_ADATA_LONG,
        C.TEST_ADATA_SHORT,
        type=str,
        required=True,
        help=C.TEST_ADATA_HELP,
    )
    parser_test.add_argument(
        C.TEST_MASKS_LONG,
        C.TEST_MASKS_SHORT,
        default=C.TEST_MASKS_DEFAULT,
        help=C.TEST_MASKS_HELP,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    if args.command == C.SUB_COMMAND_TEST:
        visualize_test(args.adata, args.masks)


if __name__ == "__main__":
    main()
