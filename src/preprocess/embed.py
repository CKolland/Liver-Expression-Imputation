"""IMPORTANT!: Has to be run in different conda environment."""

import argparse
from pathlib import Path
import warnings

import numpy as np
import scanpy as sc
import scgpt as scg
from scgpt.preprocess import Preprocessor


def embedd_adata():
    """_summary_"""


def main():

    adata_file = ""
    model_dir = Path("" + "/scGPT_human")


data_is_raw = True
filter_gene_by_counts = False
filter_cell_by_counts = False
n_bins = 51

## Preprocessing
print("Data preprocessing")
# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=filter_cell_by_counts,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=True,  # 4. whether to log1p the normalized data
    result_log1p_key="Log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=False,
    # binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    # result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)


def get_scgpt_embedding(path_to_dataset):
    adata = sc.read_h5ad(path_to_project_data + path_to_dataset)
    adata.X[adata.X < 0] = 0  ## set small negative values  to 0
    adata.layers["raw_expr"] = adata.X
    preprocessor(adata, batch_key=None)
    adata.X = adata.layers["Log1p"]
    adata.var["gene_name"] = adata.var.index
    gene_col = "gene_name"  ## match 949/978 genes in vocabulary of size 60697.
    # cell_type_key = "celltype"
    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        batch_size=64,
    )

    ### extract embedding
    embedding = embed_adata.obsm["X_scGPT"]
    print("Embeddings:")
    # print(embedding)
    print("Shape of matrix with embeddings")
    print(embed_adata.obsm["X_scGPT"].shape)
    path_to_save_embedded_dataset = (
        path_to_project_data + "/embeddings" + path_to_dataset
    )
    print(path_to_save_embedded_dataset)
    embed_adata.write_h5ad(path_to_save_embedded_dataset)
    print(embed_adata)
    # print(embed_adata.var)
    # print(embed_adata.layers['X_normed'])
    # print(embed_adata.layers['Log1p'])
    # print(embed_adata.layers['raw_expr'])


## Embed datasets to 512 dimensions
get_scgpt_embedding(path_to_dataset="/LINCS/train_dataset_dmso.h5ad")
get_scgpt_embedding(path_to_dataset="/LINCS/test_dataset_dmso.h5ad")
get_scgpt_embedding(path_to_dataset="/LINCS/train_dataset_drug.h5ad")
get_scgpt_embedding(path_to_dataset="/LINCS/test_dataset_drug.h5ad")
