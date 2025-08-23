import argparse
from pathlib import Path

import visualize._constants as C
from visualize.visualize import visualize_test, compute_metrics_on_threshold


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

    # ----------------------
    #     `threshold` command
    # ----------------------

    # Visualizes test results
    parser_threshold = subparsers.add_parser(
        C.SUB_COMMAND_TRHESHOLD, help=C.THRESHOLD_HELP
    )

    parser_test.add_argument(
        C.TRHESHOLD_ADATA_LONG,
        C.TRHESHOLD_ADATA_SHORT,
        type=str,
        required=True,
        help=C.TRHESHOLD_ADATA_HELP,
    )
    parser_test.add_argument(
        C.TRHESHOLD_MASKS_LONG,
        C.TRHESHOLD_MASKS_SHORT,
        default=C.TRHESHOLD_MASKS_DEFAULT,
        help=C.TRHESHOLD_MASKS_HELP,
    )
    parser_test.add_argument(
        C.TRHESHOLD_LONG,
        C.TRHESHOLD_SHORT,
        default=C.TRHESHOLD_DEFAULT,
        help=C.TRHESHOLD_HELP,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    if args.command == C.SUB_COMMAND_TEST:
        visualize_test(args.adata, args.masks)
    elif args.command == C.SUB_COMMAND_TRHESHOLD:
        compute_metrics_on_threshold(Path(args.adata), Path(args.masks), args.threshold)


if __name__ == "__main__":
    main()
