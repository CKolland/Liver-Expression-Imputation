import anndata as ad
import scanpy as sc

files = [
    "../output/impact_seed/sc_imp_seed_13_run_2025-07-24_15-12/model_test_run_2025-08-20_02-13/sc_imp_seed_13_test_1_2025-08-20_02-13.h5ad",
    "../output/impact_seed/sc_imp_seed_13_run_2025-07-24_15-12/model_test_run_2025-08-20_02-13/sc_imp_seed_13_test_2_2025-08-20_02-13.h5ad",
]

for idx, file in enumerate(files):
    # Load your original AnnData object
    adata = ad.read_h5ad(file)
    print(f"Loaded AnnData:\n{adata}")

    # Extract predictions from .obsm (adjust key if needed)
    preds_csr = adata.obsm["predictions"]

    # Create new AnnData object from predictions
    adata_preds = ad.AnnData(X=preds_csr)
    print(f"Created new AnnData:\n{adata_preds}")

    adata_preds.X.data[adata_preds.X.data < 0] = 0
    adata_preds.X.eliminate_zeros()

    # Normalize the data (library-size correction to 1e4 counts per cell)
    sc.pp.log1p(adata_preds)
    sc.pp.normalize_total(adata_preds, target_sum=1e4)
    print("Normalized data.")

    # Save in gzipped mode
    adata_preds.write(f"predictions_{idx + 1}_normalized.h5ad", compression="gzip")
    print(f"Saved AnnData {idx + 1}.")
