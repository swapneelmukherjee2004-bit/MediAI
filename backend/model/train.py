"""
Disease Prediction ML Model Training Script
Primary Model: RandomForest (scikit-learn) — ensemble of decision trees
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


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
# RandomForest Training
# ──────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test, model_dir):
    print("\n🔵 Training RandomForest Classifier (primary model)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n✅ RandomForest Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    save_path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(rf, save_path)
    print(f"✅ RandomForest model saved → {save_path}")

    return rf, acc, preds


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

    # ── Train RandomForest ──
    rf_model, rf_acc, rf_preds = train_random_forest(
        X_train, y_train, X_test, y_test, model_dir
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  MODEL SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    print(f"  {'RandomForest (primary)':<25} {rf_acc * 100:>9.2f}%")
    print("=" * 60)

    print("\n📋 RandomForest Classification Report (Test Data):")
    print(classification_report(y_test, rf_preds, target_names=le.classes_))

    # ── Save shared artefacts ──
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(symptoms_list, os.path.join(model_dir, "symptoms_list.pkl"))
    print("\n✅ label_encoder.pkl and symptoms_list.pkl saved.")

    return rf_model, le


if __name__ == "__main__":
    train_model()
