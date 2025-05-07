# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib

# Load dataset
df = pd.read_csv("soildata.csv")

# Define features and target
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
target = "label"

# Display basic info
print("\nDataset Shape:", df.shape)
print("\nClass Distribution:\n", df[target].value_counts())

# Scale features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42, stratify=df[target])

# Train initial model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluate initial model
y_pred = model.predict(X_test)
print(f"\nInitial Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
labels = sorted(df[target].unique())
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred, labels=labels), annot=True, fmt="d",
            xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importances
importances = model.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.show()

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), 
                           param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# Evaluate best model
y_pred_best = best_model.predict(X_test)
print(f"\nBest Model Accuracy: {accuracy_score(y_test, y_pred_best)*100:.2f}%")
print("F1 Score:", f1_score(y_test, y_pred_best, average='weighted'))

# Cross-validation score
cv_scores = cross_val_score(best_model, df[features], df[target], cv=5)
print(f"\nCross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

# Save model and scaler
joblib.dump(best_model, "rf_soil_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# --- PREDICTION BLOCK ---

# Example: Predict soil type for new input
print("\n--- Soil Type Prediction ---")
input_values = [30, 31, 20, 8, 57, 5.8, 101]  # Replace with user input if needed

# Wrap input into DataFrame
input_df = pd.DataFrame([input_values], columns=features)

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
predicted_soil = best_model.predict(scaled_input)
print(f"Predicted Soil Type: {predicted_soil[0]}")
