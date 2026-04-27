import numpy as np
import pandas as pd

np.random.seed(42)

N = 1500
male_ratio = 0.8  # 80% male, 20% female

gender = np.random.choice([1, 0], size=N, p=[male_ratio, 1 - male_ratio])


age = np.random.randint(21, 60, size=N)


income = np.random.normal(50000, 15000, size=N)
income += gender * 5000  

credit_score = np.random.normal(650, 70, size=N)


employment_years = np.clip(age - 21 + np.random.normal(0, 3, size=N), 0, None)




credit_norm = (credit_score - 300) / 550
income_norm = (income - 20000) / 80000
employment_norm = employment_years / 40


approval_prob = (
    0.5 * credit_norm +
    0.3 * income_norm +
    0.2 * employment_norm
)


bias_strength = 0.15  

approval_prob += gender * bias_strength  


approval_prob = np.clip(approval_prob, 0, 1)


loan_approved = np.random.binomial(1, approval_prob)


df = pd.DataFrame({
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "gender": gender,
    "loan_approved": loan_approved
})

df.to_csv("synthetic_loan_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())