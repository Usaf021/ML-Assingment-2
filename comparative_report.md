# Comparative Analysis of ML Algorithms

Algorithms compared:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

## Consolidated Metrics

| dataset_type   | model         |   accuracy |   precision |   recall |       f1 |
|:---------------|:--------------|-----------:|------------:|---------:|---------:|
| tabular_csv    | KNN           |   0.5      |    0.52113  | 0.5      | 0.505355 |
| tabular_csv    | Decision Tree |   0.83     |    0.831883 | 0.83     | 0.830168 |
| tabular_csv    | Naive Bayes   |   0.81     |    0.811326 | 0.81     | 0.810458 |
| image_data     | KNN           |   0.352778 |    0.467964 | 0.352778 | 0.312114 |
| image_data     | Decision Tree |   0.313889 |    0.313138 | 0.313889 | 0.312091 |
| image_data     | Naive Bayes   |   0.375    |    0.352606 | 0.375    | 0.342016 |

## Best Model for tabular_csv

- **Model:** Decision Tree
- **Accuracy:** 0.8300
- **Precision:** 0.8319
- **Recall:** 0.8300
- **F1-score:** 0.8302

## Best Model for image_data

- **Model:** Naive Bayes
- **Accuracy:** 0.3750
- **Precision:** 0.3526
- **Recall:** 0.3750
- **F1-score:** 0.3420
