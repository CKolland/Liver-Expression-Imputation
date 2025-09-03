**Master’s Thesis Project – Bioinformatics** (Goethe University Frankfurt):  
*Neural Network-Based Gene Expression Imputation for Subcellular Spatial Transcriptomics Data*

For the companion repository focusing on data exploration and preprocessing, see: [EDA-Spatial-Liver](https://github.com/CKolland/EDA-Spatial-Liver).

# STEXI

**STEXI** (Spatial Transcriptomics Expression Imputation) is a deep learning toolkit for the imputation of spatial transcriptomics data. It supports flexible preprocessing, model training, and evaluation workflows, and is designed for scalable, GPU-accelerated analysis of large single-cell and spatial omics datasets.

## ✨ Features

- 🧹 **Flexible Preprocessing:** Easily concatenate, filter, and prepare AnnData files for downstream analysis.
- 🤖 **Deep Learning Imputation:** State-of-the-art neural network models for imputing missing gene expression values in spatial transcriptomics data.
- ⚡ **Scalable Training:** Efficient training routines with support for GPU acceleration and large datasets.
- 📊 **Evaluation Tools:** Comprehensive metrics and visualization utilities for assessing imputation performance.
- 🧩 **Modular Design:** Easily extend or customize preprocessing, model, and evaluation components.

## 📂 Package Structure

```
Liver-Expression-Imputation/
├── src/
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── concat_h5ad.py
│   │   └── ...
│   ├── fit_model/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   └── ...
│   ├── test_model/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   └── ...
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── confy.py
│   │   └── ...
│   └── ...
├── pyproject.toml
├── README.md
└── ...
```

- `preprocess/`: Tools for preprocessing and managing AnnData files.
- `fit_model/`: Training routines and CLI for model fitting.
- `test_model/`: Evaluation and testing utilities.
- `utils/`: Shared utilities and configuration classes.

## ⚙️ Installation

```sh
git clone https://github.com/CKolland/Liver-Expression-Imputation
cd Liver-Expression-Imputation
pip install -e .
```

This will install STEXI in editable mode, allowing you to make changes to the source code and use the command-line tools immediately.

## 🚀 Usage

After installation, you can use the provided command-line tools:

```sh
preprocess concat -d data1.h5ad data2.h5ad -o merged.h5ad
fit -c config.yml -o output_dir
```

Refer to the documentation and CLI help (`-h` or `--help` flags) for detailed usage instructions.

## 📜 License

This project is licensed under the MIT License.  
See [LICENSE](LICENSE) for details.

## 📬 Contact

Maintainer: Christian Kolland ([Schulz Lab](https://schulzlab.github.io/))

For questions, requests (including access to large data files), or feedback, please contact the maintainer directly.
