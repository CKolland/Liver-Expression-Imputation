import argparse

import visualize._constants as C
from visualize.visualize import (
    visualize_test,
    compute_metrics_against_baseline,
    compute_metrics_on_threshold,
)


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

    # --------------------------
    #     `baseline` command
    # --------------------------

    # Visualizes test results
    parser_baseline = subparsers.add_parser(
        C.SUB_COMMAND_BASELINE, help=C.BASELINE_HELP
    )

    parser_baseline.add_argument(
        C.BASELINE_ADATA_LONG,
        C.BASELINE_ADATA_SHORT,
        type=str,
        required=True,
        help=C.BASELINE_ADATA_HELP,
    )
    parser_baseline.add_argument(
        C.BASELINE_MASKS_LONG,
        C.BASELINE_MASKS_SHORT,
        default=C.BASELINE_MASKS_DEFAULT,
        help=C.BASELINE_MASKS_HELP,
    )

    # ----------------------
    #     `threshold` command
    # ----------------------

    # Visualizes test results
    parser_threshold = subparsers.add_parser(
        C.SUB_COMMAND_THRESHOLD, help=C.THRESHOLD_HELP
    )

    parser_threshold.add_argument(
        C.THRESHOLD_ADATA_LONG,
        C.THRESHOLD_ADATA_SHORT,
        type=str,
        required=True,
        help=C.THRESHOLD_ADATA_HELP,
    )
    parser_threshold.add_argument(
        C.THRESHOLD_MASKS_LONG,
        C.THRESHOLD_MASKS_SHORT,
        default=C.THRESHOLD_MASKS_DEFAULT,
        help=C.THRESHOLD_MASKS_HELP,
    )
    parser_threshold.add_argument(
        C.THRESHOLD_LONG,
        C.THRESHOLD_SHORT,
        default=C.THRESHOLD_DEFAULT,
        help=C.THRESHOLD_HELP,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    if args.command == C.SUB_COMMAND_TEST:
        visualize_test(args.adata, args.masks)
    elif args.command == C.SUB_COMMAND_BASELINE:
        compute_metrics_against_baseline(args.adata, args.masks)
    elif args.command == C.SUB_COMMAND_THRESHOLD:
        compute_metrics_on_threshold(args.adata, args.masks, args.threshold)


if __name__ == "__main__":
    main()
