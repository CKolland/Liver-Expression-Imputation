import argparse

import anndata as ad
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str)
    parser.add_argument("--type", "-t", default=0)

    args = parser.parse_args()

    adata = ad.read_h5ad(args.file)

    if args.type == 0:
        df = pd.read_feather("./whole_transcriptome.feather")
        adata.uns["target_names"] = df[0].to_list()
    elif args.type == 1:
        df = pd.read_feather("./ST_genes.feather")
        adata.uns["target_names"] = df[0].to_list()

    adata.write(args.file, compression="gzip")


if __name__ == "__main__":
    main()
