# ML Assignment 2 — Comparative Analysis of ML Algorithms

This project compares the following algorithms on **two dataset types**:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

Dataset types:
1. Tabular CSV dataset
2. Image dataset

## Kaggle datasets selected

### 1) Tabular CSV dataset
- **Mobile Price Classification**
- Kaggle: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
- Expected target column: `price_range`
- Expected CSV file: `train.csv`

### 2) Image dataset
- **Intel Image Classification**
- Kaggle: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- Use class-folder images under: `seg_train/seg_train`

## Setup

Install dependencies, then run the script with your dataset paths.

Output files are created in `results/`:
- `metrics.csv`
- `comparative_report.md`

## Example run (with Kaggle data)

Use your own local paths after downloading the Kaggle datasets:
- `--tabular-csv`: path to Mobile Price `train.csv`
- `--tabular-target`: `price_range`
- `--image-dir`: path to Intel image folder containing class subfolders

## Quick validation mode

You can validate the full pipeline without Kaggle data using:
- `--demo`

Demo mode uses built-in scikit-learn datasets (Wine + Digits) just to verify code execution.
