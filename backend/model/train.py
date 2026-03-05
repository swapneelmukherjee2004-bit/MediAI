"""
Disease Prediction ML Model Training Script
Uses a Random Forest Classifier trained on Kaggle Disease Prediction Dataset.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def load_kaggle_data():
    """Load training and testing data from the downloaded Kaggle CSVs."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, 'data', 'Training.csv')
    test_path = os.path.join(base_dir, 'data', 'Testing.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Kaggle datasets not found in {base_dir}/data. Please download them.")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Some Kaggle datasets have an empty column at the end due to trailing commas
    if df_train.columns[-1] == 'Unnamed: 133':
        df_train = df_train.drop('Unnamed: 133', axis=1)
    if 'Unnamed: 133' in df_test.columns:
        df_test = df_test.drop('Unnamed: 133', axis=1)

    # The last column is 'prognosis'
    symptoms = list(df_train.columns[:-1])
    
    # Strip any trailing whitespace from labels
    df_train['prognosis'] = df_train['prognosis'].str.strip()
    df_test['prognosis'] = df_test['prognosis'].str.strip()

    X_train = df_train.drop('prognosis', axis=1).values
    y_train = df_train['prognosis'].values

    X_test = df_test.drop('prognosis', axis=1).values
    y_test = df_test['prognosis'].values

    return X_train, y_train, X_test, y_test, symptoms


def train_model():
    print("Loading Kaggle Dataset...")
    try:
        X_train, y_train_labels, X_test, y_test_labels, symptoms_list = load_kaggle_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Fit Label Encoder on all possible labels to ensure consistency
    le = LabelEncoder()
    le.fit(np.concatenate((y_train_labels, y_test_labels)))
    
    y_train = le.transform(y_train_labels)
    y_test = le.transform(y_test_labels)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of diseases: {len(le.classes_)}")
    print(f"Number of features (symptoms): {X_train.shape[1]}")

    # Train Random Forest
    print("\nTraining Random Forest Classifier on Real Kaggle Data...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    predictions = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, predictions)
    print(f"\nRandom Forest Test Accuracy: {rf_acc:.4f}")

    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, predictions, target_names=le.classes_))

    # Save model, label encoder, and feature names
    model_dir = os.path.dirname(__file__)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(rf_model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))
    joblib.dump(symptoms_list, os.path.join(model_dir, 'symptoms_list.pkl'))

    print(f"\n✅ Model saved to {model_dir}/model.pkl")
    print(f"✅ Label encoder saved to {model_dir}/label_encoder.pkl")
    print(f"✅ Symptoms list saved to {model_dir}/symptoms_list.pkl")

    return rf_model, le


if __name__ == "__main__":
    train_model()
