import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_digits, load_wine
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_tabular_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0,
    )

    return preprocessor


def run_tabular_experiment(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in tabular dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = build_tabular_preprocessor(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Naive Bayes": GaussianNB(),
    }

    rows: List[Dict[str, float]] = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test.to_numpy(), preds)
        rows.append({"dataset_type": "tabular_csv", "model": model_name, **metrics})

    return pd.DataFrame(rows)


def load_images_from_folders(
    image_root: Path,
    image_size: Tuple[int, int] = (32, 32),
    max_per_class: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    if not image_root.exists() or not image_root.is_dir():
        raise ValueError(f"Image directory does not exist or is not a directory: {image_root}")

    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(
            "No class subfolders found in image directory. "
            "Expected structure: image_root/class_name/image_files"
        )

    for class_dir in class_dirs:
        count = 0
        for file_path in class_dir.iterdir():
            if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            with Image.open(file_path) as img:
                img = img.convert("L").resize(image_size)
                arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

            X_list.append(arr)
            y_list.append(class_dir.name)
            count += 1

            if count >= max_per_class:
                break

    if not X_list:
        raise ValueError("No valid images found in class subfolders.")

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


def run_image_experiment(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    models = {
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Naive Bayes": GaussianNB(),
    }

    rows: List[Dict[str, float]] = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        rows.append({"dataset_type": "image_data", "model": model_name, **metrics})

    return pd.DataFrame(rows)


def make_markdown_report(results_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Comparative Analysis of ML Algorithms",
        "",
        "Algorithms compared:",
        "- K-Nearest Neighbors (KNN)",
        "- Decision Tree",
        "- Naive Bayes",
        "",
        "## Consolidated Metrics",
        "",
    ]

    lines.append(results_df.to_markdown(index=False))
    lines.append("")

    for dataset_type in results_df["dataset_type"].unique():
        subset = results_df[results_df["dataset_type"] == dataset_type]
        best_row = subset.sort_values("f1", ascending=False).iloc[0]
        lines.extend(
            [
                f"## Best Model for {dataset_type}",
                "",
                f"- **Model:** {best_row['model']}",
                f"- **Accuracy:** {best_row['accuracy']:.4f}",
                f"- **Precision:** {best_row['precision']:.4f}",
                f"- **Recall:** {best_row['recall']:.4f}",
                f"- **F1-score:** {best_row['f1']:.4f}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_demo_mode(test_size: float, random_state: int) -> pd.DataFrame:
    wine = load_wine(as_frame=True)
    tabular_df = wine.frame.copy()
    tabular_df["target"] = wine.target

    tabular_results = run_tabular_experiment(
        tabular_df,
        target_col="target",
        test_size=test_size,
        random_state=random_state,
    )

    digits = load_digits()
    image_results = run_image_experiment(
        digits.data,
        digits.target,
        test_size=test_size,
        random_state=random_state,
    )

    return pd.concat([tabular_results, image_results], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparative analysis of KNN, Decision Tree, and Naive Bayes on tabular + image data"
    )
    parser.add_argument("--tabular-csv", type=str, default=None, help="Path to tabular CSV file")
    parser.add_argument("--tabular-target", type=str, default=None, help="Target column in tabular CSV")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to image root directory")
    parser.add_argument(
        "--image-max-per-class",
        type=int,
        default=300,
        help="Maximum images per class to load",
    )
    parser.add_argument("--img-size", type=int, default=32, help="Square size for image resizing")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a built-in demo (Wine + Digits) to validate the pipeline",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        results_df = run_demo_mode(args.test_size, args.random_state)
    else:
        if not args.tabular_csv or not args.tabular_target:
            raise ValueError(
                "For non-demo mode, provide both --tabular-csv and --tabular-target."
            )
        if not args.image_dir:
            raise ValueError("For non-demo mode, provide --image-dir.")

        tabular_path = Path(args.tabular_csv)
        if not tabular_path.exists():
            raise ValueError(f"Tabular CSV not found: {tabular_path}")

        tabular_df = pd.read_csv(tabular_path)
        tabular_results = run_tabular_experiment(
            tabular_df,
            target_col=args.tabular_target,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        X_img, y_img = load_images_from_folders(
            Path(args.image_dir),
            image_size=(args.img_size, args.img_size),
            max_per_class=args.image_max_per_class,
        )
        image_results = run_image_experiment(
            X_img,
            y_img,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        results_df = pd.concat([tabular_results, image_results], ignore_index=True)

    metrics_csv = output_dir / "metrics.csv"
    report_md = output_dir / "comparative_report.md"

    results_df.to_csv(metrics_csv, index=False)
    make_markdown_report(results_df, report_md)

    print("Saved:")
    print(f"- {metrics_csv}")
    print(f"- {report_md}")
    print("\nResults:\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
