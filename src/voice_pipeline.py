"""
Voice-based Parkinson's vs. healthy classifier.

Dataset: UCI "Parkinsons" dataset (195 rows, 31 subjects).
Expected file: data/voice/parkinsons.csv
Target column: 'class' or 'status' (1 = Parkinson's, 0 = healthy) -- both are auto-detected.

Usage:
    python src/voice_pipeline.py
"""

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

DATA_PATH = Path("data/voice/parkinsons.csv")
MODEL_OUT = Path("models/voice_best_model.joblib")

# Columns that are identifiers, not features, if present in the CSV.
ID_COLUMNS = {"name", "id", "subject#", "subject", "patient_id"}
TARGET_CANDIDATES = ("class", "status", "label", "target")

CV_FOLDS = 5
RANDOM_STATE = 42

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        sys.exit(
            f"Could not find {path}.\n"
            "Download the UCI/Kaggle Parkinsons voice CSV and save it there "
            "(see README.md for links)."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_col is None:
        sys.exit(
            f"Couldn't find a target column among {TARGET_CANDIDATES} in {list(df.columns)}. "
            "Edit TARGET_CANDIDATES in this script to match your file."
        )

    y = df[target_col].astype(int)
    drop_cols = {target_col} | {c for c in ID_COLUMNS if c in df.columns}
    X = df.drop(columns=list(drop_cols))
    X = X.select_dtypes(include=[np.number])  # keep only numeric acoustic features

    majority_baseline = y.value_counts(normalize=True).max()
    print(f"Loaded {len(df)} rows, {X.shape[1]} numeric features, target column = '{target_col}'")
    print(f"Class balance:\n{y.value_counts()}")
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
    X, y = load_data(DATA_PATH)
    models = build_models()

    print(f"Running {CV_FOLDS}-fold stratified cross-validation (dataset is small, "
          f"so a single train/test split would be too noisy to trust)...\n")
    results = evaluate(X, y, models)
    print(results.to_string())

    # Refit the best model (by mean CV accuracy) on the full dataset and save it.
    accuracy_means = {
        name: float(results.loc[name, "accuracy"].split(" ")[0]) for name in results.index
    }
    best_name = max(accuracy_means, key=accuracy_means.get)
    print(f"\nBest model by mean CV accuracy: {best_name} ({accuracy_means[best_name]:.3f})")

    best_pipe = models[best_name]
    best_pipe.fit(X, y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "feature_names": list(X.columns), "model_name": best_name}, MODEL_OUT)
    print(f"Saved refit model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
