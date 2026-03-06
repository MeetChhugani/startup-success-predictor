import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- Load Model ---
artifact = joblib.load("models/startup_success_model.pkl")
model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]
encoder = artifact["encoder"]

# --- Page Config ---
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="centered"
)

# --- Title ---
st.title("🚀 AI Startup Success Predictor")
st.markdown("Enter your startup details below to predict success probability.")
st.divider()

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    company_age = st.number_input(
        "Company Age (years)",
        min_value=0,
        max_value=50,
        value=3
    )
    funding = st.number_input(
        "Total Funding (USD)",
        min_value=0,
        max_value=1000000000,
        value=500000,
        step=10000
    )
    funding_rounds = st.number_input(
        "Number of Funding Rounds",
        min_value=0,
        max_value=20,
        value=1
    )
    milestones = st.number_input(
        "Number of Milestones",
        min_value=0,
        max_value=50,
        value=2
    )
    funding_duration_days = st.number_input(
        "Funding Duration (days)",
        min_value=0,
        max_value=5000,
        value=180
    )

with col2:
    relationships = st.number_input(
        "Number of Relationships",
        min_value=0,
        max_value=200,
        value=5
    )
    investment_rounds = st.number_input(
        "Investment Rounds",
        min_value=0,
        max_value=20,
        value=1
    )
    milestone_duration_days = st.number_input(
        "Milestone Duration (days)",
        min_value=0,
        max_value=5000,
        value=180
    )
    days_to_first_funding = st.number_input(
        "Days to First Funding",
        min_value=0,
        max_value=5000,
        value=365
    )

st.divider()

# --- Categorical Inputs ---
col3, col4 = st.columns(2)

with col3:
    category = st.selectbox("Industry / Category", [
        'Unknown', 'advertising', 'analytics', 'automotive', 'biotech',
        'cleantech', 'consulting', 'design', 'ecommerce', 'education',
        'enterprise', 'fashion', 'finance', 'games_video', 'government',
        'hardware', 'health', 'hospitality', 'legal', 'local',
        'manufacturing', 'medical', 'messaging', 'mobile', 'music',
        'nanotech', 'network_hosting', 'news', 'nonprofit', 'other',
        'pets', 'photo_video', 'public_relations', 'real_estate', 'search',
        'security', 'semiconductor', 'social', 'software', 'sports',
        'transportation', 'travel', 'web'
    ])

with col4:
    country = st.selectbox("Country", [
        'USA', 'GBR', 'CAN', 'IND', 'DEU',
        'FRA', 'ESP', 'IRL', 'AUS', 'Unknown', 'Other'
    ])

st.divider()

# --- Boolean Inputs ---
col5, col6 = st.columns(2)
with col5:
    is_silicon_valley = st.checkbox("Based in Silicon Valley (USA - CA)?")
    has_website = st.checkbox("Has a Website?", value=True)
with col6:
    has_description = st.checkbox("Has a Company Description?", value=True)

st.divider()

# --- Predict Button ---
if st.button("🔍 Predict Success", use_container_width=True):

    # --- Build Input ---
    funding_per_round = funding / funding_rounds if funding_rounds > 0 else 0

    numeric_dict = {
        "company_age": company_age,
        "log_funding_total_usd": np.log1p(funding),
        "funding_rounds": funding_rounds,
        "milestones": milestones,
        "relationships": relationships,
        "investment_rounds": investment_rounds,
        "funding_duration_days": funding_duration_days,
        "milestone_duration_days": milestone_duration_days,
        "days_to_first_funding": days_to_first_funding,
        "log_funding_per_round": np.log1p(funding_per_round),
        "is_silicon_valley": int(is_silicon_valley),
        "has_website": int(has_website),
        "has_description": int(has_description)
    }

    # --- Encode Categorical ---
    cat_df = pd.DataFrame(
        [[category, country]],
        columns=['category_code', 'country_code_grouped']
    )
    encoded_cats = encoder.transform(cat_df)
    encoded_cat_df = pd.DataFrame(
        encoded_cats,
        columns=encoder.get_feature_names_out(
            ['category_code', 'country_code_grouped']
        )
    )

    # --- Combine Features ---
    numeric_df = pd.DataFrame([numeric_dict])
    input_df = pd.concat([numeric_df, encoded_cat_df], axis=1)
    input_df = input_df.reindex(columns=features, fill_value=0)

    # --- Predict ---
    prob = model.predict_proba(input_df)[0][1]
    prediction = prob >= threshold

    # --- Results ---
    st.subheader("📊 Prediction Results")

    col7, col8 = st.columns(2)
    with col7:
        st.metric(
            label="Success Probability",
            value=f"{prob * 100:.1f}%"
        )
    with col8:
        if prediction:
            st.success("✅ High Success Potential")
        else:
            st.error("❌ Low Success Potential")

    st.progress(float(prob))
    st.divider()

    if prob >= 0.7:
        st.info("💡 Strong indicators of success — high funding, good relationships and solid industry presence.")
    elif prob >= 0.4:
        st.warning("💡 Moderate potential. Consider improving funding and industry relationships.")
    else:
        st.error("💡 Weak success indicators. Significant improvements needed in funding and market presence.")

    # --- SHAP Explanation ---
    st.divider()
    st.subheader("🔍 Why This Prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns.tolist()
        ),
        max_display=10,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()