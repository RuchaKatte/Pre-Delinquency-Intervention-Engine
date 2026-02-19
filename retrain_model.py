import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("synthetic_cashpulse_data.csv")

# Scale base features
features_to_scale = [
    "income_drop_pct",
    "balance_drop_pct",
    "emi_to_income_ratio",
    "credit_utilization",
    "failed_autodebit",
    "discretionary_spend_drop"
]

scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Create Stress Indices
df["income_stability_index"] = df["income_drop_pct"]
df["liquidity_strength_index"] = df["balance_drop_pct"]

df["debt_pressure_index"] = (
    df["emi_to_income_ratio"] +
    df["credit_utilization"] +
    df["failed_autodebit"]
) / 3

df["behavioral_shift_index"] = df["discretionary_spend_drop"]

df["financial_stress_score"] = (
    0.30 * df["income_stability_index"] +
    0.25 * df["liquidity_strength_index"] +
    0.25 * df["debt_pressure_index"] +
    0.20 * df["behavioral_shift_index"]
)

# Model features
features = [
    "income_stability_index",
    "liquidity_strength_index",
    "debt_pressure_index",
    "behavioral_shift_index",
    "financial_stress_score"
]

X = df[features]
y = df["default_next_30_days"]

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X, y)

# Save model
pickle.dump(model, open("xgb_model.pkl", "wb"))

print("Model retrained and saved successfully!")
