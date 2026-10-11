import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Predictive Analysis — PRP Project 2", page_icon="🤖", layout="wide")

from utils.style import inject_css, page_header, callout, NAVY, TEAL, PURPLE, BERRY, LOW_COLOR, MEDIUM_COLOR, HIGH_COLOR
from utils.model import build_model, score_one_loan

inject_css()

page_header(
    "Predictive Analysis",
    "The flagship model — Logistic Regression, Loan Default Risk, built with smf.logit() "
    "exactly like the class case study",
    tag="PHASE 3 OF 3 — FLAGSHIP MODEL",
)

callout(
    "There is no real 'did this loan default' field anywhere in this dataset. "
    "<b>derived_default_risk_flag</b> is a constructed label — Poor credit_score (&lt;580) AND "
    "top-25% loan_amount OR interest_rate — disclosed as synthetic, never presented as an "
    "observed outcome. Fraud stays diagnostic-only (no ground truth exists to train against); "
    "this default-risk model is the project's one and only flagship predictive model.",
    kind="warn",
)

with st.spinner("Training the flagship model (statsmodels logit, cached after first run)..."):
    m = build_model()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏗️ How the Model Was Built", "📊 Performance & Validation", "💡 Live Risk Scoring", "📋 Full Loan Book"]
)

# ============================================================ TAB 1: BUILD
with tab1:
    st.markdown("#### 1. The target: a constructed label, not a real outcome")
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive class rate", f"{m['target_rate']*100:.2f}%")
    c2.metric("P75 loan_amount threshold", f"${m['p75_amount']:,.0f}")
    c3.metric("P75 interest_rate threshold", f"{m['p75_rate']:.2f}%")
    st.caption("22.18% positive — comfortably above the 20% balance guideline, so no over/under-sampling needed.")

    st.markdown("#### 2. Model 1 — all 5 candidate features")
    st.code(m["m1_summary"], language=None)
    st.caption(
        "credit_score, loan_amount, interest_rate are all extremely significant (p<0.001), exactly "
        "as expected since the target was built directly from thresholds on these three columns. "
        "tenure_at_origination_years comes back non-significant (p≈0.63) — matching the Diagnostic "
        "Analysis finding that tenure has no real relationship to anything else here."
    )

    st.markdown("#### 3. Variable reduction — Somers' D (bivariate feature importance, cutoff 0.2)")
    st.dataframe(m["somers_df"], use_container_width=True)
    st.caption(
        f"Features clearing the 0.2 cutoff: {', '.join(m['kept_features'])}. "
        f"tenure_at_origination_years scores only ~0.005 and is dropped."
    )

    st.markdown("#### 4. Multi-collinearity check — VIF")
    st.dataframe(m["vif_df"], use_container_width=True)
    callout(
        "loan_amount and loan_to_credit_ratio are both well above the usual VIF 'problem' "
        "threshold (5-10) — a mechanical side-effect: since credit_score only ranges 300-850 "
        "while loan_amount ranges into the hundreds of thousands, the ratio is driven almost "
        "entirely by loan_amount's own variation. Unlike the class case study (where p-values drove "
        "the cut), here VIF has to drive the decision: drop the redundant engineered ratio, keep "
        "the raw, directly-interpretable loan_amount.",
        kind="finding",
    )

    st.markdown("#### 5. Final model (Model 3)")
    st.code(m["m3_summary"], language=None)
    st.dataframe(m["vif_final"], use_container_width=True)
    callout(
        "<b>derived_default_risk_flag ~ credit_score + loan_amount + interest_rate</b> — all 3 "
        "highly significant (p&lt;0.001), VIF ≈1.0 for all three (collinearity resolved). Exactly the "
        "3 core features named in the original Predictive Analysis Plan.",
        kind="finding",
    )

    st.markdown("#### 6. Interpretation — odds ratios")
    c4, c5, c6 = st.columns(3)
    c4.metric("50-pt credit_score drop", f"{m['odds_credit_50']:.2f}x odds")
    c5.metric("+$10,000 loan_amount", f"{m['odds_loan_10k']:.2f}x odds")
    c6.metric("+1pp interest_rate", f"{m['odds_rate_1pt']:.2f}x odds")
    st.caption(
        "A 50-point drop in credit score roughly DOUBLES the odds of being flagged high-risk. "
        "Each extra percentage point of interest rate raises the odds by far the most — the "
        "strongest single lever in the model."
    )

# ============================================================ TAB 2: PERFORMANCE
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train AUC", f"{m['train_auc']:.3f}")
    c2.metric("Test AUC", f"{m['test_auc']:.3f}")
    c3.metric("Train Accuracy", f"{m['train_accuracy']*100:.1f}%")
    c4.metric("Test Accuracy", f"{m['test_accuracy']*100:.1f}%")
    callout(
        f"AUC of ~0.93 on BOTH train and test — near-identical, meaning no overfitting. Best "
        f"cutoff = <b>{m['best_cutoff']:.2f}</b> (vs. the class case study's 0.31 — there's no "
        f"universal 'correct' cutoff, it depends on where THIS model's sensitivity and specificity "
        f"trade off).",
        kind="finding",
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(m["scan_df"], x="proba", y="sen_spec",
                      title="Sensitivity + Specificity vs. Cutoff (chosen cutoff maximizes this)",
                      color_discrete_sequence=[NAVY])
        fig.add_vline(x=m["best_cutoff"], line_dash="dash", line_color=HIGH_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        train_pf = m["train_pf"]
        fig2 = px.histogram(train_pf, x="prob", color="actual", barmode="overlay", nbins=40,
                            opacity=0.5, title="Predicted Probability by Actual Class",
                            color_discrete_map={0: LOW_COLOR, 1: HIGH_COLOR},
                            labels={"actual": "Actual class"})
        fig2.add_vline(x=m["best_cutoff"], line_dash="dash", line_color="black")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Good separation between the two curves confirms the model is actually distinguishing the classes.")

    st.markdown("#### Decile analysis — rank-ordering validation")
    dec = m["decile_summary"].copy()
    dec.columns = ["Decile", "Min Prob", "Max Prob", "Actual High-Risk Count", "Loan Count", "% Actually High-Risk"]
    st.dataframe(dec.style.format({
        "Min Prob": "{:.3f}", "Max Prob": "{:.3f}", "% Actually High-Risk": "{:.1f}%"
    }), use_container_width=True)
    callout(
        "The riskiest decile alone captures the large majority of actual high-risk loans in that "
        "group (~88% of that decile is genuinely high-risk); the bottom deciles capture almost "
        "none. The model doesn't just classify at one cutoff — it RANKS loans in a way a credit "
        "team could act on directly ('scrutinize the top decile or two, standard process for the rest').",
        kind="finding",
    )

# ============================================================ TAB 3: LIVE SCORING
with tab3:
    st.markdown("#### Score a loan in real time")
    st.markdown(
        "Move the sliders to describe a hypothetical loan — the probability updates instantly "
        "using the fitted Model 3 coefficients (no refitting needed)."
    )

    col1, col2 = st.columns([1, 1.4])
    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650, step=1)
        loan_amount = st.number_input("Loan Amount (USD)", min_value=1000.0, max_value=300000.0,
                                       value=150000.0, step=1000.0)
        interest_rate = st.slider("Interest Rate (%)", 2.0, 15.0, 8.5, step=0.1)

        prob = score_one_loan(m["coef"], credit_score, loan_amount, interest_rate)

        if prob >= 0.5:
            tier, color = "3. High", HIGH_COLOR
        elif prob >= m["best_cutoff"]:
            tier, color = "2. Medium", MEDIUM_COLOR
        else:
            tier, color = "1. Low", LOW_COLOR

        st.markdown(
            f"""
            <div style="background-color:{color}20; border-left:6px solid {color}; padding:18px;
            border-radius:8px; margin-top:16px;">
            <h3 style="margin:0; color:{color};">Predicted probability: {prob*100:.1f}%</h3>
            <p style="margin:4px 0 0 0; font-size:1.1rem;">Risk tier: <b>{tier.split('. ')[1]}</b></p>
            <p style="margin:4px 0 0 0; color:#5F5F5F; font-size:0.85rem;">Decision cutoff: {m['best_cutoff']:.2f} (Medium) / 0.50 (High)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        grid_credit = np.linspace(300, 850, 60)
        probs = [score_one_loan(m["coef"], cs, loan_amount, interest_rate) for cs in grid_credit]
        fig = px.line(x=grid_credit, y=probs,
                      labels={"x": "Credit Score", "y": "Predicted Probability of High Risk"},
                      title=f"Sensitivity to Credit Score (loan=${loan_amount:,.0f}, rate={interest_rate}%)",
                      color_discrete_sequence=[NAVY])
        fig.add_hline(y=m["best_cutoff"], line_dash="dash", line_color=MEDIUM_COLOR)
        fig.add_hline(y=0.5, line_dash="dash", line_color=HIGH_COLOR)
        fig.add_vline(x=credit_score, line_dash="dot", line_color=BERRY)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This model was trained on a CONSTRUCTED target built from these exact same 3 columns "
            "— treat this widget as a demonstration of the model's own logic, not as real "
            "underwriting advice, since no real default outcome was ever observed in this data."
        )

# ============================================================ TAB 4: FULL BOOK
with tab4:
    book = m["scored_book"]
    tier_counts = book["risk_tier"].value_counts().sort_index()

    c1, c2, c3 = st.columns(3)
    for col, tname in zip([c1, c2, c3], ["1. Low", "2. Medium", "3. High"]):
        cnt = int(tier_counts.get(tname, 0))
        col.metric(tname.split(". ")[1], f"{cnt:,}", f"{100*cnt/len(book):.1f}%")

    fig = px.histogram(book, x="prob", nbins=40, title="Predicted Probability Distribution — Full Loan Book",
                       color_discrete_sequence=[NAVY])
    fig.add_vline(x=m["best_cutoff"], line_dash="dash", line_color=MEDIUM_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Browse scored loans")
    tier_filter = st.multiselect("Filter by risk tier", ["1. Low", "2. Medium", "3. High"],
                                  default=["3. High"])
    display_cols = ["loan_id", "customer_id", "credit_score", "loan_amount", "interest_rate",
                     "prob", "risk_tier"]
    filtered = book[book["risk_tier"].isin(tier_filter)][display_cols].sort_values("prob", ascending=False)
    st.dataframe(filtered.head(500).style.format({"loan_amount": "${:,.0f}", "interest_rate": "{:.2f}%",
                                                    "prob": "{:.3f}"}), use_container_width=True, height=400)
    st.caption(f"Showing up to 500 of {len(filtered):,} matching loans, sorted by predicted probability.")
