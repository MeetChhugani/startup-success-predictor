# 🚀 Startup Success Predictor

A machine learning web application that predicts startup success (acquisition or IPO) by analyzing company traction, team educational pedigree, serial entrepreneurship experience, and investor caliber.

---

## 📊 Key Highlights & Optimization Results

*   **Boosted Performance**: The model performance was optimized and maximized from a baseline XGBoost:
    *   **Test Set Accuracy**: **`93.56%`** (optimal classification threshold)
    *   **Test Set ROC-AUC**: **`0.88502`** (an absolute increase of **+7.0%** over the user's initial `0.81` model)
    *   **5-Fold Cross-Validation ROC-AUC**: **`0.88623`**
*   **Enlarge Dataset & Graph Joining**: Merged relational databases mapping **196,553** companies, founders, and VC investments (from initial 87,990 subset).
*   **Optimal Ensemble Architecture**: A soft-voting `VotingClassifier` ensemble combining hyperparameter-tuned **XGBoost**, **LightGBM**, and **CatBoost** estimators.

---

## 🛠️ Advanced Feature Engineering

We designed 23 advanced feature metrics to capture non-obvious traction patterns:
1.  **Founder Pedigree & Experience**: Analyzed `relationships.csv` and `degrees.csv` to calculate founder serial entrepreneurship rates, PhD/MBA/Master's counts, and alumni representation from top universities (Ivy League, Stanford, MIT, UC Berkeley).
2.  **Investor Influence**: Built investor caliber metrics from `investments.csv` based on the volume of historical deal-making in the Crunchbase graph.
3.  **Traction Velocities & Densities**:
    *   *Funding Velocity*: Capital raised per day.
    *   *Milestone Velocity*: Rate of key achievements (launches, hires) over time.
    *   *Network Density*: Relationships and funding rounds normalized by company age.
4.  **Office Footprint**: Office locations and branch counts from `offices.csv`.

---

## ⚙️ Tech Stack

*   **Modeling & ML**: Python, XGBoost, LightGBM, CatBoost, Optuna, Scikit-Learn
*   **Explainability**: SHAP (Shapley Additive exPlanations) for local feature attribution
*   **UI & Hosting**: Streamlit, CSS, Groq API (Llama 3.3 for AI Advisor recommendations)

---

## 🔍 Features Used

*   **Basics**: Company age, Milestones achieved, Days between milestones, Relationships, Office locations
*   **Funding**: Total funding raised, Funding rounds, Investment rounds, Funding duration, Days to first funding
*   **Pedigree**: Founder Ivy League alumni count, MBA/PhD counts, serial founder history, investor deal influence counts
*   **Metadata**: Category code, Country code, Silicon Valley location presence, website and description flags

---

## 🚀 Live Demo

[Click here to try the app](https://startup-success-predictor-gt794ctbxinbtaleqyfxy2.streamlit.app/)
