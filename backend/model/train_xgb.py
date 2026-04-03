"""
XGBoost DART training script — runs as a separate subprocess to avoid
PyTorch + XGBoost runtime conflict on Python 3.14.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


def load_data():
    """Load all data and apply a reproducible 80/20 stratified train/test split."""
    from sklearn.model_selection import train_test_split

    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, "data", "Training.csv")
    test_path  = os.path.join(base_dir, "data", "Testing.csv")

    # Merge both CSVs into one unified dataset
    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)

    if "Unnamed: 133" in df.columns:
        df.drop("Unnamed: 133", axis=1, inplace=True)

    df["prognosis"] = df["prognosis"].str.strip()

    X = df.drop("prognosis", axis=1).values.astype(np.float32)
    y = df["prognosis"].values

    # 80 / 20 stratified split — reproducible via random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, y_train, X_test, y_test


def main():
    print("\n🟢 Training XGBoost DART Classifier (secondary model)...")
    X_train, y_train_labels, X_test, y_test_labels = load_data()

    le = LabelEncoder()
    le.fit(list(y_train_labels) + list(y_test_labels))
    y_train = le.transform(y_train_labels)
    y_test  = le.transform(y_test_labels)

    xgb = XGBClassifier(
        booster="dart",
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        rate_drop=0.1,
        skip_drop=0.5,
        sample_type="uniform",
        normalize_type="tree",
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        verbosity=0,
        seed=42,
        n_jobs=1,
    )
    xgb.fit(X_train, y_train)

    preds = xgb.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n✅ XGBoost DART Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    model_dir = os.path.dirname(__file__)
    save_path = os.path.join(model_dir, "xgb_dart_model.pkl")
    joblib.dump(xgb, save_path)
    print(f"✅ XGBoost DART model saved → {save_path}")

    # Write accuracy result for parent process to read
    result_path = os.path.join(model_dir, "_xgb_acc.json")
    with open(result_path, "w") as f:
        json.dump({"accuracy": acc}, f)


if __name__ == "__main__":
    main()
