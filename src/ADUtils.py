import argparse
import os

import anndata as ad
import numpy as np


def main():
    """
    Main entry point for AnnData preprocessing utilities.

    This function sets up a command-line interface (CLI) for performing common preprocessing tasks on AnnData objects,
    such as adding a log1p-transformed layer or concatenating multiple AnnData files.

    **Subcommands:**

    - ``scale``:
        Adds a log1p-transformed layer to the AnnData object if it does not already exist.

        **Arguments**:
            - ``--adata``, ``-d`` (``str``, required): Path to the input AnnData (.h5ad) file.

        **Behavior**:
            - If the "log1p" layer is not present, computes ``np.log1p(adata.X)`` and adds it as a new layer.
            - Saves the modified AnnData object to a new file with "_log1p.h5ad" suffix.
            - Prints a message indicating the result.

    - ``concat``:
        Concatenates multiple AnnData objects into a single AnnData object.

        **Arguments**:
            - ``--adatas``, ``-a`` (``str``, nargs="+", required): List of paths to AnnData (.h5ad) files to concatenate.
            - ``--output``, ``-o`` (``str``, default="./"): Output path or directory for the concatenated AnnData object.

        **Behavior**:
            - Concatenates the provided AnnData objects along the observation axis.
            - Assigns a "batch" label to each input object using its filename as the key.
            - Saves the concatenated object to the specified output path (or as "concatenated.h5ad" if a directory is given).
            - Prints a message indicating the output location.

    **Usage**:
        .. code-block:: bash

            python preprocess_data.py scale --adata path/to/file.h5ad
            python preprocess_data.py concat --adatas file1.h5ad file2.h5ad --output output_dir/

    :raises SystemExit: If required arguments are missing or invalid.
    """
    # --------------------
    #     Setup parser
    # --------------------
    parser = argparse.ArgumentParser(description="AnnData utils")

    # Create a subparser object for multiple CLI commands
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="AUtils subcommands"
    )

    # --------------------
    #     Scale command
    # --------------------
    # Adds log1p-transformed data to AnnData
    parser_scale = subparsers.add_parser(
        "scale", help="Add log1p layer to AnnData object if not present"
    )
    parser_scale.add_argument(
        "--adata",
        "-d",
        type=str,
        required=True,
        help="Path to the AnnData object file",
    )

    # --------------------
    #     Concat command
    # --------------------
    # Merges multiple AnnData files into one
    parser_concat = subparsers.add_parser(
        "concat", help="Concatenate multiple AnnData objects"
    )
    parser_concat.add_argument(
        "--adatas",
        "-d",
        type=str,
        nargs="+",
        required=True,
        help="Paths to AnnData object files to concatenate",
    )
    parser_concat.add_argument(
        "--output",
        "-o",
        type=str,
        default="./",
        help="Path to save the concatenated AnnData object",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # --------------------
    #     Scale logic
    # --------------------
    if args.command == "scale":
        adata_path = args.adata

        # Load the AnnData file
        adata = ad.read_h5ad(adata_path)

        # Only add log1p layer if it doesn't already exist
        if "log1p" not in adata.layers:
            adata.layers["log1p"] = np.log1p(adata.X)

            # Save with new filename indicating log1p transformation
            out_path = os.path.splitext(adata_path)[0] + "_log1p.h5ad"
            adata.write(out_path)
            print(f"Added log1p layer and saved to {out_path}")
        else:
            # If already present, do nothing
            print("log1p layer already present. No changes made.")

    # --------------------
    #     Concat logic
    # --------------------
    elif args.command == "concat":
        # Load all input AnnData objects
        adatas = [ad.read_h5ad(p) for p in args.adatas]

        # Concatenate them along the observation (rows) axis
        adata_concat = ad.concat(
            adatas,
            join="outer",
            label="batch",  # Assigns a 'batch' column to identify source
            keys=[
                os.path.basename(p) for p in args.adatas
            ],  # Use filenames as batch labels
        )

        # Determine if output is a directory or file path
        output_path = args.output
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "concatenated.h5ad")

        # Save the merged AnnData object
        adata_concat.write(output_path)
        print(f"Concatenated AnnData objects saved to {output_path}")


if __name__ == "__main__":
    main()
