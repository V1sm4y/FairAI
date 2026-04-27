import joblib
import numpy as np

# Load model
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example user input
# [age, income, credit_score, employment_years, gender]
import pandas as pd

user = pd.DataFrame([{
    "age": 35,
    "income": 75000,
    "credit_score": 680,
    "employment_years": 7,
    "gender": 0
}])

user_scaled = scaler.transform(user)


pred = model.predict(user_scaled)[0]
prob = model.predict_proba(user_scaled)[0][1]

# Output
if pred == 1:
    print(f"Loan Approved ✅ (Confidence: {prob:.2f})")
else:
    print(f"Loan Rejected ❌ (Confidence: {prob:.2f})")