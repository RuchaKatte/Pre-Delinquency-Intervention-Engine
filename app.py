import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="CashPulse", layout="wide")

st.title("💳 CashPulse: Pre-Delinquency Intelligence Engine")

# -----------------------------
# Load Data & Model
# -----------------------------
df = pd.read_csv("synthetic_cashpulse_data.csv")
model = pickle.load(open("xgb_model.pkl", "rb"))

# -----------------------------
# Generate Basic Customer Info
# -----------------------------
df["customer_name"] = "Customer_" + df["customer_id"].astype(str)
df["email"] = df["customer_id"].apply(lambda x: f"customer{x}@bank.com")
df["account_created"] = "2022-01-01"

# -----------------------------
# Preprocess Data
# -----------------------------
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

features = [
    "income_stability_index",
    "liquidity_strength_index",
    "debt_pressure_index",
    "behavioral_shift_index",
    "financial_stress_score"
]

# -----------------------------
# Sidebar Portfolio Insights
# -----------------------------
st.sidebar.title("📊 Portfolio Insights")

portfolio_probs = []
for i in df["customer_id"]:
    temp = df[df["customer_id"] == i]
    prob = model.predict_proba(temp[features])[0][1]
    portfolio_probs.append(prob)

df["portfolio_risk"] = portfolio_probs

st.sidebar.metric("Total Customers", len(df))
st.sidebar.metric("High Risk Customers", sum(df["portfolio_risk"] > 0.85))

# -----------------------------
# Customer Selection
# -----------------------------
customer_id = st.selectbox(
    "Select Customer ID",
    df["customer_id"].unique()
)

customer_data = df[df["customer_id"] == customer_id]

# -----------------------------
# Customer Profile Section
# -----------------------------
st.markdown("## 👤 Customer Profile")

colA, colB, colC = st.columns(3)

colA.write(f"**Name:** {customer_data['customer_name'].values[0]}")
colB.write(f"**Email:** {customer_data['email'].values[0]}")
colC.write(f"**Account Created:** {customer_data['account_created'].values[0]}")

# -----------------------------
# Stress Breakdown
# -----------------------------
st.markdown("## 📉 4-Dimensional Financial Stress Breakdown")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Income Stability",
            round(float(customer_data["income_stability_index"]), 3))

col2.metric("Liquidity Strength",
            round(float(customer_data["liquidity_strength_index"]), 3))

col3.metric("Debt Pressure",
            round(float(customer_data["debt_pressure_index"]), 3))

col4.metric("Behavioral Shift",
            round(float(customer_data["behavioral_shift_index"]), 3))

# -----------------------------
# Risk Prediction
# -----------------------------
X = customer_data[features]
risk_prob = model.predict_proba(X)[0][1]

st.markdown("## 📊 Risk Assessment")
st.metric("Predicted Default Probability (30 Days)",
          round(risk_prob, 3))

# -----------------------------
# Risk Band & Intervention
# -----------------------------
if risk_prob < 0.4:
    st.success("Stable")
    intervention = "Monitor"
elif risk_prob < 0.7:
    st.warning("Early Stress")
    intervention = "Soft Reminder + Financial Health Check"
elif risk_prob < 0.85:
    st.error("High Risk")
    intervention = "Offer EMI Restructuring"
else:
    st.error("🚨 Pre-Delinquency Alert")
    intervention = "Offer Payment Holiday (Customer Consent Required)"

st.markdown("## 🎯 Recommended Intervention")
st.info(intervention)

# -----------------------------
# Portfolio Risk Distribution
# -----------------------------
st.markdown("## 📈 Portfolio Risk Distribution")

fig_hist = plt.figure()
plt.hist(df["portfolio_risk"], bins=20)
plt.xlabel("Risk Probability")
plt.ylabel("Number of Customers")
st.pyplot(fig_hist)

# -----------------------------
# Stress Radar Chart
# -----------------------------
st.markdown("## 🧠 Stress Radar Overview")

labels = ["Income", "Liquidity", "Debt", "Behavior"]

values = [
    float(customer_data["income_stability_index"]),
    float(customer_data["liquidity_strength_index"]),
    float(customer_data["debt_pressure_index"]),
    float(customer_data["behavioral_shift_index"])
]

values += values[:1]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig_radar = plt.figure()
ax = fig_radar.add_subplot(111, polar=True)
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.3)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
st.pyplot(fig_radar)

# -----------------------------
# Interactive Simulation
# -----------------------------
st.sidebar.title("🔬 Simulate Stress Scenario")

sim_income = st.sidebar.slider("Income Stability", 0.0, 1.0,
                               float(customer_data["income_stability_index"]))

sim_liq = st.sidebar.slider("Liquidity Strength", 0.0, 1.0,
                            float(customer_data["liquidity_strength_index"]))

sim_debt = st.sidebar.slider("Debt Pressure", 0.0, 1.0,
                             float(customer_data["debt_pressure_index"]))

sim_beh = st.sidebar.slider("Behavioral Shift", 0.0, 1.0,
                            float(customer_data["behavioral_shift_index"]))

sim_score = 0.30*sim_income + 0.25*sim_liq + 0.25*sim_debt + 0.20*sim_beh

sim_df = pd.DataFrame([[sim_income, sim_liq, sim_debt, sim_beh, sim_score]],
                      columns=features)

sim_prob = model.predict_proba(sim_df)[0][1]

st.sidebar.metric("Simulated Risk", round(sim_prob, 3))

# -----------------------------
# SHAP Explainability
# -----------------------------
st.markdown("## 🔍 Explainability (Why is this customer risky?)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

fig_shap = plt.figure()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X.iloc[0],
        feature_names=features
    ),
    show=False
)

st.pyplot(fig_shap)
