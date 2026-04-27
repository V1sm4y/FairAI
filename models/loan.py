import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("synthetic_loan_data.csv")

X = df.drop(columns=["loan_approved"])
y = df["loan_approved"]
A = df["gender"]

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, A, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# SCALE
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred = model.predict(X_test_scaled)

# -----------------------------
# OVERALL PERFORMANCE
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.3f}")

# -----------------------------
# FAIRNESS FUNCTION
# -----------------------------
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return tpr, fpr

# -----------------------------
# GROUP METRICS
# -----------------------------
print("\n--- Fairness by Gender ---")

results = {}

for group in [0, 1]:
    mask = (A_test == group)

    y_true_g = y_test[mask]
    y_pred_g = y_pred[mask]

    tpr, fpr = compute_metrics(y_true_g, y_pred_g)

    label = "Female (0)" if group == 0 else "Male (1)"
    print(f"\n{label}:")
    print(f"TPR: {tpr:.3f}")
    print(f"FPR: {fpr:.3f}")

    results[group] = (tpr, fpr)

# -----------------------------
# FAIRNESS GAP
# -----------------------------
tpr_diff = abs(results[1][0] - results[0][0])
fpr_diff = abs(results[1][1] - results[0][1])

print("\n--- Fairness Gap ---")
print(f"TPR Difference: {tpr_diff:.3f}")
print(f"FPR Difference: {fpr_diff:.3f}")

# -----------------------------
# SAVE MODEL + SCALER
# -----------------------------
joblib.dump(model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")