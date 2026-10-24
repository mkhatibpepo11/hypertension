"""Train and evaluate a reproducible educational hypertension classifier."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATASET = "electricsheepafrica/africa-synth-hypertension-hypertension-cvd-dataset-all"
TARGET = "hypertension_status"
DROP = ["systolic_bp_mmhg", "diastolic_bp_mmhg", "bp_controlled", "diagnosed", "on_medication", "medication_adherence", "patient_id"]
THRESHOLD = 0.30  # From the original project; not clinically validated.


def prepare(df):
    df = df.copy().drop_duplicates()
    if TARGET not in df:
        raise ValueError(f"Missing target: {TARGET}")
    y = df[TARGET].astype(str).str.strip().str.lower().map({"normal": 0, "hypertensive": 1})
    if y.isna().any():
        raise ValueError("Unexpected or missing target labels")
    X = df.drop(columns=[TARGET, *DROP], errors="ignore")
    return X, y.astype(int)


def metrics(y, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def main():
    dataset = load_dataset(
        DATASET,
        data_files={
            "train": "hypertension_cvd_africa_extra_large_10000.csv",
            "validation": "hypertension_cvd_africa_large_5000.csv",
            "test": "hypertension_cvd_africa_test_2000.csv",
        },
    )
    required = ("train", "validation", "test")
    if not all(split in dataset for split in required):
        raise ValueError(f"Expected splits: {required}")
    X_train, y_train = prepare(dataset["train"].to_pandas())
    X_val, y_val = prepare(dataset["validation"].to_pandas())
    X_test, y_test = prepare(dataset["test"].to_pandas())
    # All feature selection and imputation statistics are learned on train only.
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    transformer = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    pipeline = Pipeline([
        ("preprocess", transformer),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=10,
                                          random_state=42, class_weight="balanced_subsample")),
    ])
    pipeline.fit(X_train, y_train)
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    # Test is evaluated once, after choosing a threshold. 0.30 is inherited
    # from the original notebook; this is not an optimized clinical cutoff.
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    report = {
        "dataset": DATASET,
        "note": "Synthetic-data educational model. Not for diagnosis or clinical decisions.",
        "split_sizes_after_dedup": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "feature_columns": X_train.columns.tolist(),
        "validation_at_0_50": metrics(y_val, val_prob, 0.50),
        "validation_at_0_30": metrics(y_val, val_prob, THRESHOLD),
        "test_at_0_30": metrics(y_test, test_prob, THRESHOLD),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": X_train.columns.tolist(), "threshold": THRESHOLD},
                ROOT / "models" / "model.joblib")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"validation_auc": report["validation_at_0_30"]["roc_auc"],
                      "test_auc": report["test_at_0_30"]["roc_auc"]}, indent=2))
    print("Saved models/model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
