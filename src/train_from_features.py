import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def train_model():
    csv_path = PROJECT_ROOT / "data" / "urfd_extracted_features.csv"
    
    print(f"Loading extracted features from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Display class distribution
    print("\nClass distribution in dataset:")
    print(df['label'].value_counts())

    # Separate features and labels
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")

    # Train Random Forest Model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ADL (0)", "Fall (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save trained model
    model_filename = PROJECT_ROOT / "models" / "fall_detection_model.pkl"
    model_filename.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_filename)
    print(f"\nSaved trained model to '{model_filename}'.")

if __name__ == "__main__":
    train_model()