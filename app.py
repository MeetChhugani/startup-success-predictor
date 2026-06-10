import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import os
from groq import Groq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Progress stepper */
.stepper{display:flex;align-items:center;gap:0;margin-bottom:2rem}
.s-step{display:flex;align-items:center;gap:8px;font-size:13px;color:#888;flex:1}
.s-step.active{color:#111;font-weight:600}
.s-step.done{color:#1D9E75}
.s-dot{width:28px;height:28px;border-radius:50%;border:1.5px solid #ccc;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;flex-shrink:0;background:#fff}
.s-step.active .s-dot{background:#1D9E75;border-color:#1D9E75;color:#fff}
.s-step.done .s-dot{background:#E1F5EE;border-color:#1D9E75;color:#0F6E56}
.s-line{flex:1;height:1px;background:#e0e0e0;margin:0 4px}

/* Benchmark chips */
.bench-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;margin-bottom:12px}
.bench-chip{background:#f5f5f5;border:1px solid #e0e0e0;border-radius:6px;padding:4px 10px;font-size:11px;color:#666;cursor:pointer;display:inline-block}

/* SHAP bars */
.shap-wrap{margin-bottom:8px}
.shap-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.shap-name{font-size:13px;color:#666;width:170px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.shap-track{flex:1;height:8px;background:#f0f0f0;border-radius:100px;overflow:hidden}
.shap-pos{height:100%;border-radius:100px;background:#1D9E75}
.shap-neg{height:100%;border-radius:100px;background:#D85A30;margin-left:auto}
.shap-val{font-size:12px;color:#888;width:44px;text-align:right;flex-shrink:0}

/* Verdict box */
.verdict-success{background:#E1F5EE;border:1px solid #9FE1CB;border-radius:10px;padding:14px 18px;margin:12px 0}
.verdict-fail{background:#FAECE7;border:1px solid #F5C4B3;border-radius:10px;padding:14px 18px;margin:12px 0}
.verdict-title{font-size:16px;font-weight:600;margin-bottom:4px}
.verdict-sub{font-size:13px;color:#555}

/* Section label */
.sec-label{font-size:11px;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#999;margin-bottom:6px}

/* Toggle cards */
.toggle-hint{font-size:12px;color:#888;margin-top:4px}

/* Metric cards */
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}
.metric-card{background:#f9f9f9;border-radius:8px;padding:14px;text-align:center}
.metric-val{font-size:22px;font-weight:600;color:#111}
.metric-val.green{color:#1D9E75}
.metric-lbl{font-size:11px;color:#888;margin-top:4px}

/* Progress bar */
.prob-track{height:10px;background:#f0f0f0;border-radius:100px;overflow:hidden;margin:8px 0}
.prob-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,#9FE1CB,#1D9E75)}
.prob-fail{background:linear-gradient(90deg,#F5C4B3,#D85A30)}
.prob-labels{display:flex;justify-content:space-between;font-size:11px;color:#888;margin-top:4px}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/startup_success_model.pkl")

artifact = load_model()
model    = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]
encoder  = artifact["encoder"]

CATEGORIES = [
    'Unknown','advertising','analytics','automotive','biotech','cleantech',
    'consulting','design','ecommerce','education','enterprise','fashion',
    'finance','games_video','government','hardware','health','hospitality',
    'legal','local','manufacturing','medical','messaging','mobile','music',
    'nanotech','network_hosting','news','nonprofit','other','pets',
    'photo_video','public_relations','real_estate','search','security',
    'semiconductor','social','software','sports','transportation','travel','web',
]
COUNTRIES = ['USA','GBR','CAN','IND','DEU','FRA','ESP','IRL','AUS','Unknown','Other']

# ── Session state defaults ────────────────────────────────────────────────────
def _defaults():
    defaults = dict(
        step=1,
        company_age=3, milestones=2, relationships=5,
        milestone_duration_days=180,
        funding=500_000, funding_rounds=2, investment_rounds=1,
        funding_duration_days=180, days_to_first_funding=365,
        category="software", country="USA",
        is_silicon_valley=False, has_website=True, has_description=True,
        result=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_defaults()


# ── Stepper HTML ──────────────────────────────────────────────────────────────
def render_stepper(current_step):
    steps = ["Basics", "Funding", "Results"]
    html = '<div class="stepper">'
    for i, label in enumerate(steps, 1):
        cls = "done" if i < current_step else ("active" if i == current_step else "")
        dot = "✓" if i < current_step else str(i)
        html += f'<div class="s-step {cls}"><div class="s-dot">{dot}</div><span>{label}</span></div>'
        if i < len(steps):
            html += '<div class="s-line"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── STEP 1: Basics ────────────────────────────────────────────────────────────
def step_basics():
    render_stepper(1)
    st.markdown("### Tell us about your startup")
    st.caption("Basic company info and traction signals.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.company_age = st.slider(
            "Company age (years)", 0, 20, st.session_state.company_age)
        st.caption("Median at first raise: 2–4 years")

        st.session_state.milestones = st.slider(
            "Milestones achieved", 0, 30, st.session_state.milestones)
        st.caption("Product launches, partnerships, key hires")

    with col2:
        st.session_state.relationships = st.slider(
            "Network relationships", 0, 100, st.session_state.relationships)
        st.caption("Advisors, investors, strategic contacts")

        st.session_state.milestone_duration_days = st.slider(
            "Days between milestones", 0, 1000, st.session_state.milestone_duration_days)
        st.caption("Avg time between key achievements")

    st.divider()
    st.markdown('<div class="sec-label">Startup signals</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.is_silicon_valley = st.checkbox(
            "🏙️ Silicon Valley", value=st.session_state.is_silicon_valley)
        st.markdown('<div class="toggle-hint">Based in SF Bay Area</div>', unsafe_allow_html=True)
    with c2:
        st.session_state.has_website = st.checkbox(
            "🌐 Has website", value=st.session_state.has_website)
        st.markdown('<div class="toggle-hint">Active online presence</div>', unsafe_allow_html=True)
    with c3:
        st.session_state.has_description = st.checkbox(
            "📄 Has description", value=st.session_state.has_description)
        st.markdown('<div class="toggle-hint">Public company profile</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("Continue to Funding →", use_container_width=True, type="primary"):
        st.session_state.step = 2
        st.rerun()


# ── STEP 2: Funding ───────────────────────────────────────────────────────────
def step_funding():
    render_stepper(2)
    st.markdown("### Funding & industry")
    st.caption("Capital raised, round history, and market segment.")
    st.divider()

    # Funding slider with benchmark chips
    st.markdown("**Total funding raised**")
    funding = st.slider(
        "Total funding (USD)", 0, 50_000_000,
        st.session_state.funding, step=50_000,
        label_visibility="collapsed",
        format="$%d",
    )
    st.session_state.funding = funding

    benchmarks = {
        "Pre-seed $0": 0,
        "Seed $500K": 500_000,
        "Series A $3M": 3_000_000,
        "Series B $15M": 15_000_000,
        "Series C $50M+": 50_000_000,
    }
    chips_html = '<div class="bench-row">'
    for label in benchmarks:
        chips_html += f'<span class="bench-chip">{label}</span>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    if funding >= 1_000_000:
        st.caption(f"Selected: **${funding/1_000_000:.1f}M**")
    elif funding >= 1_000:
        st.caption(f"Selected: **${funding//1_000:,}K**")
    else:
        st.caption(f"Selected: **${funding:,}**")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.funding_rounds = st.slider(
            "Funding rounds", 0, 10, st.session_state.funding_rounds)
        st.caption("Median at seed: 1–2 rounds")

        st.session_state.funding_duration_days = st.slider(
            "Funding duration (days)", 0, 2000, st.session_state.funding_duration_days)
        st.caption("First to last funding date")

    with col2:
        st.session_state.investment_rounds = st.slider(
            "Investment rounds", 0, 10, st.session_state.investment_rounds)
        st.caption("Distinct investor groups")

        st.session_state.days_to_first_funding = st.slider(
            "Days to first funding", 0, 2000, st.session_state.days_to_first_funding)
        st.caption("From founding to first raise")

    st.divider()
    st.markdown('<div class="sec-label">Industry & location</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.session_state.category = st.selectbox(
            "Industry category", CATEGORIES,
            index=CATEGORIES.index(st.session_state.category))
    with col4:
        st.session_state.country = st.selectbox(
            "Country", COUNTRIES,
            index=COUNTRIES.index(st.session_state.country))

    st.divider()
    col_back, col_next = st.columns([1, 2])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Predict success →", use_container_width=True, type="primary"):
            st.session_state.result = run_prediction()
            st.session_state.step = 3
            st.rerun()


# ── Prediction logic ──────────────────────────────────────────────────────────
def run_prediction():
    s = st.session_state
    funding_per_round = s.funding / s.funding_rounds if s.funding_rounds > 0 else 0

    numeric_dict = {
        "company_age":              s.company_age,
        "log_funding_total_usd":    np.log1p(s.funding),
        "funding_rounds":           s.funding_rounds,
        "milestones":               s.milestones,
        "relationships":            s.relationships,
        "investment_rounds":        s.investment_rounds,
        "funding_duration_days":    s.funding_duration_days,
        "milestone_duration_days":  s.milestone_duration_days,
        "days_to_first_funding":    s.days_to_first_funding,
        "log_funding_per_round":    np.log1p(funding_per_round),
        "is_silicon_valley":        int(s.is_silicon_valley),
        "has_website":              int(s.has_website),
        "has_description":          int(s.has_description),
    }

    cat_df = pd.DataFrame(
        [[s.category, s.country]],
        columns=["category_code", "country_code_grouped"],
    )
    encoded = encoder.transform(cat_df)
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(["category_code", "country_code_grouped"])
    )

    input_df = pd.concat([pd.DataFrame([numeric_dict]), encoded_df], axis=1)
    input_df = input_df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]
    success = prob >= threshold

    # SHAP values
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    shap_series = pd.Series(sv, index=input_df.columns)

    return dict(prob=prob, success=success, input_df=input_df, shap_series=shap_series)


# ── STEP 3: Results ───────────────────────────────────────────────────────────
def step_results():
    render_stepper(3)
    r = st.session_state.result
    if r is None:
        st.warning("No prediction found. Go back and fill in your details.")
        if st.button("← Start over"):
            st.session_state.step = 1
            st.rerun()
        return

    prob    = r["prob"]
    success = r["success"]
    shap_s  = r["shap_series"]

    # ── Metrics row ──
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-val {'green' if success else ''}">{prob*100:.1f}%</div>
        <div class="metric-lbl">Success probability</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">0.81</div>
        <div class="metric-lbl">Model ROC-AUC</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">87,990</div>
        <div class="metric-lbl">Companies in training data</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Verdict ──
    if success:
        st.markdown(f"""
        <div class="verdict-success">
          <div class="verdict-title" style="color:#0F6E56">✅ High success potential</div>
          <div class="verdict-sub">Your probability ({prob*100:.1f}%) clears the {threshold*100:.0f}% decision threshold.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-fail">
          <div class="verdict-title" style="color:#993C1D">⚠️ Below success threshold</div>
          <div class="verdict-sub">Your probability ({prob*100:.1f}%) is below the {threshold*100:.0f}% threshold. See improvement tips below.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability bar ──
    fill_class = "prob-fill" if success else "prob-fill prob-fail"
    st.markdown(f"""
    <div class="prob-track">
      <div class="{fill_class}" style="width:{prob*100:.1f}%"></div>
    </div>
    <div class="prob-labels">
      <span>Low</span>
      <span>Threshold {threshold*100:.0f}%</span>
      <span>High 100%</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── SHAP explanation ──
    st.markdown("#### Why this prediction?")
    top_pos = shap_s.nlargest(5)
    top_neg = shap_s.nsmallest(5)
    max_abs = shap_s.abs().max()

    tab_pos, tab_neg = st.tabs(["✅ Boosters", "⚠️ Risk factors"])

    with tab_pos:
        if max_abs > 0:
            rows_html = ""
            for feat, val in top_pos.items():
                pct = abs(val) / max_abs * 100
                label = feat.replace("_", " ").replace("log ", "").replace("category code ", "").replace("country code grouped ", "")
                rows_html += f"""
                <div class="shap-row">
                  <div class="shap-name">{label}</div>
                  <div class="shap-track"><div class="shap-pos" style="width:{pct:.0f}%"></div></div>
                  <div class="shap-val">+{val:.2f}</div>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)

    with tab_neg:
        if max_abs > 0:
            rows_html = ""
            for feat, val in top_neg.items():
                pct = abs(val) / max_abs * 100
                label = feat.replace("_", " ").replace("log ", "").replace("category code ", "").replace("country code grouped ", "")
                rows_html += f"""
                <div class="shap-row">
                  <div class="shap-name">{label}</div>
                  <div class="shap-track"><div class="shap-neg" style="width:{pct:.0f}%"></div></div>
                  <div class="shap-val" style="color:#993C1D">{val:.2f}</div>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)

    st.divider()

    # ── Groq improvement tips ──
    st.markdown("#### Get improvement tips")
    if st.button("💡 Ask AI advisor for tips →", use_container_width=True):
        with st.spinner("Analysing your startup profile..."):
            try:
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                s = st.session_state
                top_neg_str = "\n".join([
                    f"- {k.replace('_',' ')}: SHAP = {v:.3f}"
                    for k, v in top_neg.items()
                ])
                prompt = f"""
You are a VC analyst. A startup has been assessed by an ML model with these details:
- Success probability: {prob*100:.1f}%
- Verdict: {"High potential" if success else "Below threshold"}
- Industry: {s.category}, Country: {s.country}
- Funding raised: ${s.funding:,}, Rounds: {s.funding_rounds}
- Milestones: {s.milestones}, Relationships: {s.relationships}
- Days to first funding: {s.days_to_first_funding}

Top negative SHAP factors dragging down the score:
{top_neg_str}

Give exactly 3 specific, actionable improvement tips based on the negative factors.
Be direct and concrete. Format as numbered list. Max 3 sentences each.
"""
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                )
                tips = response.choices[0].message.content
                st.markdown("**AI advisor recommendations:**")
                st.markdown(tips)
            except Exception as e:
                st.error(f"Could not reach AI advisor: {e}")

    st.divider()
    col_back, col_reset = st.columns(2)
    with col_back:
        if st.button("← Edit inputs", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_reset:
        if st.button("🔄 Start new prediction", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────
STEP_FNS = {1: step_basics, 2: step_funding, 3: step_results}
STEP_FNS[st.session_state.step]()