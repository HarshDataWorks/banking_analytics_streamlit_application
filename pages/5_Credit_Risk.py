import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Credit Risk — PRP Project 2", page_icon="💳", layout="wide")

from utils.style import inject_css, page_header, callout, NAVY, TEAL, PURPLE, BERRY, LOW_COLOR, MEDIUM_COLOR, HIGH_COLOR
from utils.data_loader import load_customers
from utils.model import build_model

inject_css()

page_header(
    "Dashboard 2 — Credit Risk",
    "Stakeholder: Credit Risk Team · Loan portfolio health + the flagship default-risk model's output",
    tag="DASHBOARD 2 OF 4",
)

with st.spinner("Loading model output (cached after first run)..."):
    m = build_model()

book = m["scored_book"]
customers = load_customers()
book = book.merge(customers[["customer_id", "credit_band"]], on="customer_id", how="left")

# ---------------------------------------------------------------- Slicers
st.sidebar.markdown("### Filters")
band_opts = sorted(book["credit_band"].dropna().unique().tolist())
sel_band = st.sidebar.multiselect("Credit Band", band_opts, default=band_opts)
tier_opts = ["1. Low", "2. Medium", "3. High"]
sel_tier = st.sidebar.multiselect("Risk Tier", tier_opts, default=tier_opts)
min_a, max_a = float(book["loan_amount"].min()), float(book["loan_amount"].max())
amt_range = st.sidebar.slider("Loan Amount Range", min_a, max_a, (min_a, max_a))
min_d, max_d = book["start_date"].min(), book["start_date"].max()
date_range = st.sidebar.slider("Origination Date Range", min_value=min_d.to_pydatetime(),
                                max_value=max_d.to_pydatetime(), value=(min_d.to_pydatetime(), max_d.to_pydatetime()))

f = book[
    book["credit_band"].isin(sel_band)
    & book["risk_tier"].isin(sel_tier)
    & book["loan_amount"].between(amt_range[0], amt_range[1])
    & book["start_date"].between(date_range[0], date_range[1])
]

# ---------------------------------------------------------------- KPIs
total_loan_book = f["loan_amount"].sum()
avg_rate = f["interest_rate"].mean() if len(f) else 0
pct_high = 100 * (f["risk_tier"] == "3. High").sum() / len(f) if len(f) else 0
avg_prob = f["prob"].mean() if len(f) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Loan Book", f"${total_loan_book/1e9:.2f}B" if total_loan_book >= 1e9 else f"${total_loan_book/1e6:.1f}M")
c2.metric("Avg Interest Rate", f"{avg_rate:.2f}%")
c3.metric("% High Risk", f"{pct_high:.1f}%")
c4.metric("Avg Predicted Probability", f"{avg_prob*100:.1f}%")

st.divider()

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(f.sample(min(8000, len(f)), random_state=1) if len(f) else f,
                     x="credit_score", y="loan_amount", color="risk_tier", opacity=0.4,
                     title="Credit Score vs. Loan Amount, colored by Risk Tier",
                     color_discrete_map={"1. Low": LOW_COLOR, "2. Medium": MEDIUM_COLOR, "3. High": HIGH_COLOR})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Shapeless, evenly-scattered cloud — credit_score and loan_amount are statistically "
        "independent (r ≈ -0.0004). The coloring shows the real structure: High/Medium tier dots "
        "cluster toward lower credit scores, since the model's decision is driven mostly by "
        "credit_score, not loan_amount."
    )

with col2:
    f = f.copy()
    f["prob_bucket"] = (f["prob"] * 10).apply(lambda x: int(x) / 10 if pd.notna(x) else None)
    bucket_counts = f["prob_bucket"].value_counts().sort_index()
    fig2 = px.bar(x=bucket_counts.index, y=bucket_counts.values,
                  labels={"x": "Predicted Probability Bucket", "y": "Loan Count"},
                  title="Predicted Default Probability Distribution", color_discrete_sequence=[NAVY])
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    tier_counts = f["risk_tier"].value_counts().reindex(tier_opts).fillna(0)
    fig3 = px.bar(x=[t.split(". ")[1] for t in tier_counts.index], y=tier_counts.values,
                  title="Risk Tier Counts", color=[t.split(". ")[1] for t in tier_counts.index],
                  color_discrete_map={"Low": LOW_COLOR, "Medium": MEDIUM_COLOR, "High": HIGH_COLOR})
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    f["origination_month"] = f["start_date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    orig = f.groupby("origination_month").size().reset_index(name="loans_originated")
    fig4 = px.line(orig, x="origination_month", y="loans_originated",
                   title="Loan Originations Over Time", color_discrete_sequence=[TEAL])
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.markdown("#### Customer Loan Detail (drill-through)")
cust_search = st.text_input("Enter a customer_id to see all of their loans", "")
if cust_search:
    detail = book[book["customer_id"].str.contains(cust_search, case=False, na=False)]
    st.dataframe(
        detail[["customer_id", "credit_score", "loan_id", "loan_amount", "interest_rate", "prob", "risk_tier"]]
        .style.format({"loan_amount": "${:,.0f}", "interest_rate": "{:.2f}%", "prob": "{:.3f}"}),
        use_container_width=True,
    )

callout(
    "Risk Tier reuses the model's own 0.21 decision cutoff (chosen by maximizing sensitivity + "
    "specificity) and a 0.50 second line for 'more likely than not to default.' "
    "derived_default_risk_flag is a constructed label, not an observed outcome — see Predictive "
    "Analysis and Data Quality & Limitations.",
)
