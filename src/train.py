import pandas as pd
import joblib
from src.models import train_rf, evaluate_model

# Load processed data
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train = train[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_train = train['label']

X_test = test[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_test = test['label']

# Train Random Forest
model = train_rf(X_train, y_train)

# Evaluate
acc, report, cm = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {acc}")
print(report)
print("Confusion Matrix:\n", cm)

# Save trained model
joblib.dump(model, "models/rf_crop_model.pkl")
print("Model saved at models/rf_crop_model.pkl")
