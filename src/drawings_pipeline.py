"""
Drawing-based Parkinson's vs. healthy classifier (spiral or wave images).

Expected folder layout (matches the Kaggle "parkinsons-drawings" dataset):
    data/drawings/<spiral|wave>/training/healthy/*.png
    data/drawings/<spiral|wave>/training/parkinson/*.png
    data/drawings/<spiral|wave>/testing/healthy/*.png
    data/drawings/<spiral|wave>/testing/parkinson/*.png

Usage:
    python src/drawings_pipeline.py --test spiral
    python src/drawings_pipeline.py --test wave
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURE_NAMES, extract_features

DATA_ROOT = Path("data/drawings")
CV_FOLDS = 5
RANDOM_STATE = 42
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

LABELS = {"healthy": 0, "parkinson": 1}


def load_dataset(shape: str) -> tuple[pd.DataFrame, pd.Series]:
    root = DATA_ROOT / shape
    if not root.exists():
        sys.exit(
            f"Could not find {root}.\n"
            "Download the Kaggle 'parkinsons-drawings' dataset and unzip it under "
            "data/drawings/ (see README.md)."
        )

    rows, labels = [], []
    for split in ("training", "testing"):  # merge both -- we do our own CV
        for label_name, label_value in LABELS.items():
            folder = root / split / label_name
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("*")):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                try:
                    feats = extract_features(str(img_path))
                except ValueError as e:
                    print(f"  skipping unreadable file: {img_path} ({e})")
                    continue
                rows.append(feats)
                labels.append(label_value)

    if not rows:
        sys.exit(f"No images found under {root} -- check the folder layout in README.md.")

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels, name="label")
    majority_baseline = y.value_counts(normalize=True).max()
    print(f"Loaded {len(X)} '{shape}' images. Class balance:\n{y.value_counts()}")
    print(f"Majority-class baseline accuracy (predicting the most common class every time): "
          f"{majority_baseline:.3f}\n")
    return X, y


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
        ),
        "Random Forest": Pipeline(
            [("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]
        ),
        "SVM (RBF)": Pipeline(
            [("scale", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]
        ),
        "MLP (neural net)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate(X: pd.DataFrame, y: pd.Series, models: dict[str, Pipeline]) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append(
            {
                "model": name,
                **{
                    metric: f"{scores[f'test_{metric}'].mean():.3f} "
                    f"(+/- {scores[f'test_{metric}'].std():.3f})"
                    for metric in SCORING
                },
            }
        )
    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["spiral", "wave"], default="spiral")
    args = parser.parse_args()

    X, y = load_dataset(args.test)
    models = build_models()

    print(f"Running {CV_FOLDS}-fold stratified cross-validation on '{args.test}' drawings...\n")
    results = evaluate(X, y, models)
    print(results.to_string())

    accuracy_means = {
        name: float(results.loc[name, "accuracy"].split(" ")[0]) for name in results.index
    }
    best_name = max(accuracy_means, key=accuracy_means.get)
    print(f"\nBest model by mean CV accuracy: {best_name} ({accuracy_means[best_name]:.3f})")

    best_pipe = models[best_name]
    best_pipe.fit(X, y)
    out_path = Path(f"models/drawings_{args.test}_best_model.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipe, "feature_names": FEATURE_NAMES, "model_name": best_name, "shape": args.test},
        out_path,
    )
    print(f"Saved refit model to {out_path}")


if __name__ == "__main__":
    main()
