import argparse

import preprocess.constants as C
from preprocess.concat_h5ad import concat_h5ad
from preprocess.create_testset import create_testset
from preprocess.scale_data import scale_data


def main():
    """
    Main entry point for AnnData preprocessing utilities.

    This function sets up a command-line interface (CLI) for performing common preprocessing tasks on AnnData objects,
    such as adding a log1p-transformed layer or concatenating multiple AnnData files.

    **Subcommands:**

    - ``concat``:
        Concatenates multiple AnnData objects into a single AnnData object.

        **Arguments**:
            - ``--adatas``, ``-a`` (``str``, nargs="+", required): List of paths to AnnData (.h5ad) files to concatenate.
            - ``--output``, ``-o`` (``str``, default="./"): Output path or directory for the concatenated AnnData object.

        **Behavior**:
            - Concatenates the provided AnnData objects along the observation axis.
            - Saves the concatenated object to the specified output path (or as "adata_concat.h5ad" if a directory is given).

    - ``scale``:
        Adds a log1p-transformed layer to the AnnData object if it does not already exist.

        **Arguments**:
            - ``--adata``, ``-d`` (``str``, required): Path to the input AnnData (.h5ad) file.

        **Behavior**:
            - If the "log1p" layer is not present, computes ``np.log1p(adata.X)`` and adds it as a new layer.
            - Saves the modified AnnData object to a new file with "_log1p.h5ad" suffix.
            - Prints a message indicating the result.

    **Usage**:
        .. code-block:: bash

            python preprocess_data.py scale --adata path/to/file.h5ad
            python preprocess_data.py concat --adatas file1.h5ad file2.h5ad --output output_dir/

    :raises SystemExit: If required arguments are missing or invalid.
    """
    # --------------------
    #     Setup parser
    # --------------------
    parser = argparse.ArgumentParser(description=C.PARSER_PREP_DESC)

    # Create a subparser object for multiple CLI commands
    subparsers = parser.add_subparsers(
        dest=C.SUBPARSERS_PREP_DEST,
        required=True,
    )

    # ------------------------
    #     `concat` command
    # ------------------------

    # Merges multiple AnnData files into one
    parser_concat = subparsers.add_parser(C.SUB_COMMAND_CONCAT, help=C.CONCAT_HELP)

    parser_concat.add_argument(
        C.CONCAT_ADATAS_LONG,
        C.CONCAT_ADATAS_SHORT,
        type=str,
        nargs="+",
        required=True,
        help=C.CONCAT_ADATAS_HELP,
    )
    parser_concat.add_argument(
        C.CONCAT_OUTPUT_LONG,
        C.CONCAT_OUTPUT_SHORT,
        type=str,
        default=C.CONCAT_OUTPUT_DEFAULT,
        help=C.CONCAT_OUTPUT_HELP,
    )

    # -----------------------
    #     `scale` command
    # -----------------------

    # Adds a new layer with scaled data to an AnnData object
    parser_scale = subparsers.add_parser(C.SUB_COMMAND_SCALE, help=C.SCALE_HELP)

    parser_scale.add_argument(
        C.SCALE_ADATA_LONG,
        C.SCALE_ADATA_SHORT,
        type=str,
        required=True,
        help=C.SCALE_ADATA_HELP,
    )
    parser_scale.add_argument(
        C.SCALE_MODE_LONG,
        C.SCALE_MODE_SHORT,
        type=str,
        choices=C.SCALE_MODE_CHOICES,
        default=C.SCALE_MODE_DEFAULT,
        help=C.SCALE_MODE_HELP,
    )
    parser_scale.add_argument(
        C.SCALE_LAYER_LONG,
        C.SCALE_LAYER_SHORT,
        type=str,
        default=C.SCALE_LAYER_DEFAULT,
        help=C.SCALE_LAYER_HELP,
    )

    # ------------------------
    #     `subset` command
    # ------------------------

    # Creates training and testing split from data
    parser_subset = subparsers.add_parser(C.SUB_COMMAND_SUBSET, help=C.SUBSET_HELP)

    parser_subset.add_argument(
        C.SUBSET_ADATA_LONG,
        C.SUBSET_ADATA_SHORT,
        type=str,
        required=True,
        help=C.SUBSET_ADATA_HELP,
    )
    parser_subset.add_argument(
        C.SUBSET_SPLIT_LONG,
        C.SUBSET_SPLIT_SHORT,
        type=float,
        default=0.1,
        help=C.SUBSET_SPLIT_HELP,
    )

    # Parse command-line arguments
    args = parser.parse_args()

    if args.command == C.SUB_COMMAND_CONCAT:
        concat_h5ad(args.adatas, out_path=args.output)
    elif args.command == C.SUB_COMMAND_SCALE:
        scale_data(args.adata, scale_mode=args.mode, data_layer=args.layer)
    elif args.command == C.SUB_COMMAND_SUBSET:
        create_testset(args.adata, split=args.split)


if __name__ == "__main__":
    main()
