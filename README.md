# CashPulse: Pre-Delinquency Intelligence Engine

**An Explainable Machine Learning System for Early Detection of Financial Distress and Proactive Banking Interventions**

CashPulse is a machine learning-based early-warning system designed to identify bank customers exhibiting potential signs of financial distress before they become delinquent. It analyzes financial and behavioural indicators to estimate the probability of default within the next 30 days and recommends appropriate preventive interventions.

The system combines engineered financial stress indicators, an XGBoost classification model, and SHAP-based explainability to provide interpretable customer-level risk assessments.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Model](#machine-learning-model)
- [Explainable AI](#explainable-ai)
- [Risk Classification and Intervention](#risk-classification-and-intervention)
- [Dashboard Features](#dashboard-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

---

## Problem Statement

Traditional banking risk assessment often focuses on identifying customers after they have missed payments or entered delinquency.

However, changes in financial behaviour can indicate potential financial stress much earlier.

Examples include:

- Sudden reductions in monthly income.
- Declining account balances.
- Increasing credit utilization.
- Failed automatic EMI deductions.
- Reductions in discretionary spending.

CashPulse explores how these financial and behavioural signals can be combined to identify customers who may be approaching delinquency.

Instead of relying solely on reactive intervention, the system aims to support proactive financial risk monitoring.

## Objectives

1. Identify early indicators of customer financial distress.
2. Construct interpretable financial stress indices from raw financial attributes.
3. Predict the probability of default within the next 30 days using machine learning.
4. Explain individual predictions using SHAP.
5. Categorize customers into different financial risk levels.
6. Recommend preventive interventions based on predicted risk.
7. Provide portfolio-level and customer-level insights through an interactive dashboard.

---

## System Architecture

The system follows a data-driven pipeline:

```text
       Customer Financial Data
                 |
                 v
       Data Preprocessing
                 |
                 v
         Feature Scaling
                 |
                 v
      Financial Stress Indices
                 |
                 v
       Financial Stress Score
                 |
                 v
        XGBoost Classifier
                 |
                 v
       Default Probability
                 |
                 v
       Risk Categorization
                 |
                 v
     Recommended Intervention
                 |
                 v
       SHAP Explainability
                 |
                 v
       Streamlit Dashboard
```

---

## Methodology

### 1. Data Preprocessing

The system uses a synthetic banking dataset containing customer-level financial and behavioural attributes.

The preprocessing pipeline includes:

- Loading structured customer data using Pandas.
- Identifying relevant financial attributes.
- Normalizing selected numerical features using MinMaxScaler.
- Constructing derived financial indicators.

### 2. Financial Stress Index Construction

Rather than directly using all raw attributes, CashPulse constructs four financial stress indicators.

| Index | Description |
|---|---|
| Income Stability Index | Captures changes in customer income |
| Liquidity Strength Index | Represents changes in account balances |
| Debt Pressure Index | Combines EMI-to-income ratio, credit utilization and failed auto-debits |
| Behavioral Shift Index | Captures changes in discretionary spending |

The Debt Pressure Index is calculated as:

```python
debt_pressure_index = (
    emi_to_income_ratio +
    credit_utilization +
    failed_autodebit
) / 3
```

The system then calculates a composite Financial Stress Score:

```python
financial_stress_score = (
    0.30 * income_stability_index +
    0.25 * liquidity_strength_index +
    0.25 * debt_pressure_index +
    0.20 * behavioral_shift_index
)
```

The weighted score combines multiple dimensions of financial behaviour into a single indicator.

These weights represent the current implementation's design choices and are not established banking risk standards.

---

## Machine Learning Model

CashPulse uses **XGBoost (Extreme Gradient Boosting)** for binary classification.

The model predicts whether a customer is likely to default within the next 30 days.

**Target Variable:**

```python
default_next_30_days
```

**Model Features:**

- Income Stability Index
- Liquidity Strength Index
- Debt Pressure Index
- Behavioral Shift Index
- Financial Stress Score

### Model Configuration

The current implementation uses:

| Parameter | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Number of Estimators | 100 |
| Maximum Depth | 4 |
| Learning Rate | 0.1 |
| Evaluation Metric | Log Loss |

The trained model is serialized using Pickle and loaded by the Streamlit application for inference.

---

## Explainable AI

A key component of CashPulse is model interpretability.

The system integrates **SHAP (SHapley Additive exPlanations)** to explain individual model predictions.

SHAP helps identify how different financial stress indicators contribute to a particular prediction.

The dashboard includes a SHAP waterfall plot to visualize feature-level contributions to an individual customer's predicted risk.

This provides additional insight into model behaviour beyond the final predicted probability.

---

## Risk Classification and Intervention

The predicted default probability is mapped into four risk categories.

| Predicted Probability | Risk Category | Suggested Intervention |
|---|---|---|
| Below 0.40 | Stable | Monitor |
| 0.40–0.69 | Early Stress | Soft Reminder + Financial Health Check |
| 0.70–0.84 | High Risk | Offer EMI Restructuring |
| 0.85 and above | Pre-Delinquency Alert | Offer Payment Holiday (Customer Consent Required) |

These thresholds and interventions are configurable rules in the current implementation. They have not been validated as optimal banking policies.

---

## Dashboard Features

CashPulse provides an interactive dashboard built using Streamlit.

### 1. Portfolio Overview
- Total number of customers.
- Number of customers classified as high risk.
- Portfolio-level risk distribution.

### 2. Customer Profile
- Customer identification.
- Basic customer information.
- Individual financial risk assessment.

### 3. Financial Stress Breakdown

Displays the four financial stress indices:

- Income Stability.
- Liquidity Strength.
- Debt Pressure.
- Behavioral Shift.

### 4. Risk Assessment

Displays the predicted probability of default within the next 30 days.

### 5. Intervention Recommendation

Provides a suggested preventive action based on the assigned risk category.

### 6. Explainability Dashboard

Uses SHAP waterfall visualization to explain individual predictions.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Model Serialization | Pickle |

---

## Project Structure

```text
Pre-Delinquency-Intervention-Engine/
│
├── app.py
│   └── Streamlit dashboard and prediction interface
│
├── retrain_model.py
│   └── Model training and serialization
│
├── stress_index_py.py
│   └── Financial stress index construction
│
├── synthetic_cashpulse_data.csv
│   └── Synthetic customer financial dataset
│
├── synthetic_cashpulse_data_with_stress.csv
│   └── Dataset containing engineered stress indicators
│
├── xgb_model.pkl
│   └── Serialized XGBoost model
│
├── requirements.txt
│   └── Project dependencies
│
├── LICENSE
│
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.9 or later
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/RuchaKatte/Pre-Delinquency-Intervention-Engine.git

cd Pre-Delinquency-Intervention-Engine
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at the local Streamlit URL displayed in the terminal.

---

## Limitations

The current implementation has certain limitations:

- The system uses synthetic banking data rather than real-world banking records.
- The financial stress indices use manually assigned weights.
- The model's predictive performance requires systematic evaluation on independent test data.
- The risk thresholds and recommended interventions are predefined.
- Generalization to real banking environments has not been established.

These limitations should be considered when interpreting the model's predictions.

---

## Future Improvements

Potential directions for further development include:

- Evaluating additional classification algorithms and comparing their performance.
- Incorporating temporal financial behaviour to identify changing risk patterns.
- Optimizing financial stress indicators through empirical validation.
- Improving model calibration and probability estimation.
- Investigating fairness and bias in financial risk prediction.
- Integrating real-world banking datasets where access and authorization permit.
- Exploring explainability-driven intervention strategies.

---

## Disclaimer

CashPulse is an experimental machine learning project developed for educational and research exploration.

It uses synthetic financial data and is not intended for actual banking decisions, credit assessments, or automated financial interventions.

All predictions and recommendations are illustrative and require further validation before any real-world application.
