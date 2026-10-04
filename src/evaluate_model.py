import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load dataset and model
df = pd.read_csv(PROJECT_ROOT / "data" / "urfd_extracted_features.csv")
X = df.drop(columns=["label"])
y = df["label"]

# Train/Test split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = joblib.load(PROJECT_ROOT / "models" / "fall_detection_model.pkl")

# 2. Generate Predictions
y_pred = model.predict(X_test)

# 3. Print Performance Report
print("=================== CLASSIFICATION REPORT ===================")
print(classification_report(y_test, y_pred, target_names=["ADL (Normal)", "Fall"]))

# 4. Save Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["ADL", "Fall"], yticklabels=["ADL", "Fall"])
plt.title("URFD Fall Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
output_path = PROJECT_ROOT / "outputs" / "confusion_matrix.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
print("Saved confusion matrix plot as 'confusion_matrix.png'")