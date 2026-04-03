"""
Disease Prediction ML Model Training Script
Primary Model:   TabNet  (pytorch-tabnet) — attention-based deep learning for tabular data
Secondary Model: XGBoost DART — gradient boosting with dropout regularisation
"""
import json
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# ────────────────────────────────────────────────────────────────
# NOTE: XGBoost DART is trained in a *separate subprocess* via
# train_xgb.py to avoid a Python 3.14 runtime conflict between
# PyTorch (TabNet) and XGBoost C extensions in the same process.
# ────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_kaggle_data():
    """Load all data and apply a reproducible 80/20 stratified train/test split."""
    from sklearn.model_selection import train_test_split

    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, "data", "Training.csv")
    test_path  = os.path.join(base_dir, "data", "Testing.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Kaggle datasets not found in {base_dir}/data. Please download them."
        )

    # Merge both CSVs into one unified dataset
    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)

    if "Unnamed: 133" in df.columns:
        df.drop("Unnamed: 133", axis=1, inplace=True)

    df["prognosis"] = df["prognosis"].str.strip()

    symptoms = list(df.columns[:-1])   # everything except 'prognosis'
    X = df.drop("prognosis", axis=1).values.astype(np.float32)
    y = df["prognosis"].values

    # 80 / 20 stratified split — reproducible via random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, y_train, X_test, y_test, symptoms


# ──────────────────────────────────────────────
# TabNet Training
# ──────────────────────────────────────────────

def train_tabnet(X_train, y_train, X_test, y_test, model_dir):
    from pytorch_tabnet.tab_model import TabNetClassifier

    print("\n🔵 Training TabNet Classifier (primary model)...")
    tabnet = TabNetClassifier(
        n_d=32,                  # width of decision step embedding
        n_a=32,                  # width of attention embedding
        n_steps=5,               # number of sequential attention steps
        gamma=1.5,               # coefficient for feature reuse
        momentum=0.02,
        mask_type="sparsemax",   # sparse attention → interpretable feature masks
        optimizer_fn=__import__("torch").optim.Adam,
        optimizer_params={"lr": 2e-2},
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=__import__("torch").optim.lr_scheduler.StepLR,
        verbose=10,
        seed=42,
    )

    tabnet.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=["test"],
        eval_metric=["accuracy"],
        max_epochs=200,
        patience=30,             # early stopping
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    preds = tabnet.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n✅ TabNet Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    # TabNet saves as a zip bundle
    save_path = os.path.join(model_dir, "tabnet_model")
    tabnet.save_model(save_path)   # creates tabnet_model.zip
    print(f"✅ TabNet model saved → {save_path}.zip")

    return tabnet, acc, preds




def train_model():
    print("=" * 60)
    print("  MediAI — Disease Prediction Model Training")
    print("=" * 60)

    print("\n📂 Loading Kaggle dataset...")
    try:
        X_train, y_train_labels, X_test, y_test_labels, symptoms_list = load_kaggle_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Encode labels
    le = LabelEncoder()
    le.fit(np.concatenate((y_train_labels, y_test_labels)))
    y_train = le.transform(y_train_labels)
    y_test  = le.transform(y_test_labels)

    print(f"\nDataset summary (80/20 stratified split):")
    print(f"  Training samples : {len(X_train)} (80%)")
    print(f"  Test samples     : {len(X_test)} (20%)")
    print(f"  Diseases (classes): {len(le.classes_)}")
    print(f"  Symptom features : {X_train.shape[1]}")

    model_dir = os.path.dirname(__file__)
    os.makedirs(model_dir, exist_ok=True)

    # ── Train both models ──
    tabnet_model, tabnet_acc, tabnet_preds = train_tabnet(
        X_train, y_train, X_test, y_test, model_dir
    )

    # XGBoost DART — runs in its own subprocess to avoid PyTorch runtime conflict
    print("\n🟢 Launching XGBoost DART training subprocess...")
    xgb_script = os.path.join(model_dir, "train_xgb.py")
    result = subprocess.run(
        [sys.executable, xgb_script],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print("❌ XGBoost DART training subprocess failed.")
        xgb_acc = 0.0
    else:
        acc_file = os.path.join(model_dir, "_xgb_acc.json")
        if os.path.exists(acc_file):
            with open(acc_file) as f:
                xgb_acc = json.load(f)["accuracy"]
            os.remove(acc_file)
        else:
            xgb_acc = 0.0

    # ── Comparison table ──
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    print(f"  {'TabNet (primary)':<25} {tabnet_acc * 100:>9.2f}%")
    print(f"  {'XGBoost DART (secondary)':<25} {xgb_acc * 100:>9.2f}%")
    print("=" * 60)

    print("\n📋 TabNet Classification Report (Test Data):")
    print(classification_report(y_test, tabnet_preds, target_names=le.classes_))

    # ── Save shared artefacts ──
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(symptoms_list, os.path.join(model_dir, "symptoms_list.pkl"))
    print("\n✅ label_encoder.pkl and symptoms_list.pkl saved.")

    return tabnet_model, le


if __name__ == "__main__":
    train_model()
