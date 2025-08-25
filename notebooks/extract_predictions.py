import anndata as ad
import pandas as pd

# Load your AnnData object
adata = ad.read_h5ad(
    "../output/impact_seed/sc_imp_seed_13_run_2025-07-24_15-12/model_test_run_2025-08-20_02-13/sc_imp_seed_13_test_1_2025-08-20_02-13.h5ad",
    backed=True,
)

# Extract predictions from .obsm (replace "predictions" with your actual key)
preds_csr = adata.obsm["predictions"]

# Convert csr_matrix to dense DataFrame
preds_df = pd.DataFrame(
    preds_csr.toarray(),
)

# Save to CSV
preds_df.to_csv("scRNA-seq_predictions.csv")
