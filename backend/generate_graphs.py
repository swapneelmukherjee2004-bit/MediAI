import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic styling
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e2e", "figure.facecolor": "#1e1e2e", "axes.edgecolor": "#1e1e2e"})

def load_data():
    base_dir = os.path.dirname(__file__)
    test_path = os.path.join(base_dir, 'data', 'Testing.csv')
    df_test = pd.read_csv(test_path)

    if 'Unnamed: 133' in df_test.columns:
        df_test = df_test.drop('Unnamed: 133', axis=1)

    df_test['prognosis'] = df_test['prognosis'].str.strip()

    X_test = df_test.drop('prognosis', axis=1).values
    y_test_labels = df_test['prognosis'].values

    return X_test, y_test_labels

def main():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'model', 'model.pkl')
    le_path = os.path.join(base_dir, 'model', 'label_encoder.pkl')
    symptoms_path = os.path.join(base_dir, 'model', 'symptoms_list.pkl')

    print("Loading model and data...")
    rf_model = joblib.load(model_path)
    le = joblib.load(le_path)
    symptoms_list = joblib.load(symptoms_path)

    X_test, y_test_labels = load_data()
    y_test = le.transform(y_test_labels)
    
    print("Making predictions...")
    predictions = rf_model.predict(X_test)

    artifact_dir = os.path.join(base_dir, 'graphs')
    os.makedirs(artifact_dir, exist_ok=True)
    print("Generating Feature Importance Graph...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 15
    top_indices = indices[:top_n]
    top_symptoms = [symptoms_list[i].replace('_', ' ').title() for i in top_indices]
    top_importances = importances[top_indices]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_symptoms, palette="viridis")
    plt.title('Top 15 Most Important Symptoms for Disease Prediction', fontsize=16, color='white')
    plt.xlabel('Importance Score', fontsize=12, color='white')
    plt.ylabel('Symptom', fontsize=12, color='white')
    plt.tick_params(colors='white', which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # 2. Performance Metrics Bar Graph (Precision, Recall, F1)
    print("Generating Performance Metrics Graph...")
    report = classification_report(y_test, predictions, target_names=le.classes_, output_dict=True)
    
    # Extract macro avg metrics
    metrics = ['precision', 'recall', 'f1-score']
    scores = [report['macro avg'][m] for m in metrics]

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=['Precision', 'Recall', 'F1-Score'], y=scores, palette="magma")
    plt.title('Overall Model Performance Metrics (Macro Average)', fontsize=16, color='white')
    plt.ylabel('Score (0 to 1)', fontsize=12, color='white')
    plt.ylim(0.9, 1.01) # Usually this Kaggle dataset gives ~100% accuracy, scaling to show differences
    plt.tick_params(colors='white')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, color='white', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'model_performance.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # 3. Confusion Matrix Picture
    # For 41 classes, it's a huge matrix, we just plot it with small text.
    print("Generating Confusion Matrix Graph...")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, cbar=False)
    plt.title('Confusion Matrix for 41 Diseases', fontsize=20, color='white')
    plt.xlabel('Predicted Label', fontsize=14, color='white')
    plt.ylabel('True Label', fontsize=14, color='white')
    plt.xticks(rotation=90, fontsize=8, color='white')
    plt.yticks(rotation=0, fontsize=8, color='white')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print("All graphs generated successfully!")

if __name__ == "__main__":
    main()
