# 🚀 Startup Success Predictor

A machine learning web app that predicts whether a startup will succeed based on key business metrics.

## 🔍 About
- Trained on 87,990 companies from the Crunchbase dataset
- XGBoost model with ROC-AUC of 0.81
- SMOTE applied to handle class imbalance (94:6 ratio)
- SHAP explainability to show why a prediction was made
- Identified survivorship bias in dataset as key finding

## 🛠️ Tech Stack
Python, XGBoost, SHAP, Scikit-learn, Streamlit, Pandas, NumPy

## 📊 Features Used
- Company age, funding amount, funding rounds
- Milestones, relationships, investment rounds
- Industry category, country, Silicon Valley location
- Funding duration, days to first funding

## 🚀 Live Demo
[Click here to try the app](https://startup-success-predictor-gt794ctbxinbtaleqyfxy2.streamlit.app/)
