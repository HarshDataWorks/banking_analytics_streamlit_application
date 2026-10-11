import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Executive Overview — PRP Project 2", page_icon="📈", layout="wide")

from utils.style import inject_css, page_header, callout, NAVY, TEAL, PURPLE, BERRY
from utils.data_loader import load_customers, load_accounts, load_loans
from utils.fraud import compute_fraud_scores
from utils.model import build_model

inject_css()

page_header(
    "Dashboard 4 — Executive Overview",
    "Stakeholder: Leadership · One page, real-time view of overall bank health — deliberately "
    "minimal, links out to the other 3 dashboards for detail",
    tag="DASHBOARD 4 OF 4",
)

customers = load_customers()
accounts = load_accounts()
loans = load_loans()

st.sidebar.markdown("### Filter")
min_d = min(accounts["open_date"].min(), loans["start_date"].min())
max_d = max(accounts["open_date"].max(), loans["start_date"].max())
date_range = st.sidebar.slider("Date Range", min_value=min_d.to_pydatetime(), max_value=max_d.to_pydatetime(),
                                value=(min_d.to_pydatetime(), max_d.to_pydatetime()))
st.sidebar.caption(
    "Per the approved plan, this is the ONLY slicer on this page — the 'fast decision-making, "
    "no digging' dashboard. Branch/risk-tier/merchant slicers live on the operational dashboards."
)

acc_f = accounts[accounts["open_date"].between(date_range[0], date_range[1])]
loans_f = loans[loans["start_date"].between(date_range[0], date_range[1])]

with st.spinner("Loading fraud + model output (cached after first run)..."):
    tx = compute_fraud_scores()
    m = build_model()

tx_f = tx[tx["transaction_date"].between(date_range[0], date_range[1])]

total_customers = len(customers)
total_aum = acc_f["balance_usd"].sum()
total_loan_book = loans_f["loan_amount"].sum()
fraud_pct = 100 * (tx_f["fraud_tier"] != "Low").sum() / len(tx_f) if len(tx_f) else 0
default_rate = 100 * (m["scored_book"]["risk_tier"] == "3. High").mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Total AUM", f"${total_aum/1e9:.2f}B" if total_aum >= 1e9 else f"${total_aum/1e6:.0f}M")
c3.metric("Total Loan Book", f"${total_loan_book/1e9:.2f}B" if total_loan_book >= 1e9 else f"${total_loan_book/1e6:.0f}M")
c4.metric("Overall Fraud-Flagged %", f"{fraud_pct:.2f}%")
c5.metric("Overall Predicted Default Rate", f"{default_rate:.1f}%")
st.caption(
    "Fraud-Flagged % here is a plain share of transaction COUNT; Dashboard 3's '% of Volume "
    "Flagged' weights by dollar amount instead — two genuinely different, both-correct numbers "
    "answering slightly different questions."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    deposits_by_month = acc_f.groupby("open_month")["balance_usd"].sum().reset_index()
    deposits_by_month["series"] = "Total AUM (by open month)"
    loans_by_month = loans_f.groupby("origination_month")["loan_amount"].sum().reset_index()
    loans_by_month["series"] = "Total Loan Book (by origination month)"
    combined = pd.concat([
        deposits_by_month.rename(columns={"open_month": "month", "balance_usd": "value"}),
        loans_by_month.rename(columns={"origination_month": "month", "loan_amount": "value"}),
    ])
    fig = px.line(combined, x="month", y="value", color="series", title="Deposits vs. Loans Trend",
                  color_discrete_sequence=[NAVY, BERRY])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    top_cities = customers["city"].value_counts().head(20).sort_values()
    fig2 = px.bar(x=top_cities.values, y=top_cities.index, orientation="h",
                 title="Geographic Distribution — Top 20 Customer Cities (not a map, by design)",
                 color_discrete_sequence=[TEAL])
    st.plotly_chart(fig2, use_container_width=True)
    callout(
        "Deliberately a bar chart, not a map: customer cities are near-unique (25,169 distinct "
        "across 50,000 customers, largest city only ~51) and branch city/country are synthetic/"
        "borrowed with zero real correspondence to the branch. A map would either show 25,000+ "
        "scattered pinpoints with no clustering, or actively mislead. See Data Quality & Limitations.",
        kind="warn",
    )

col3, col4 = st.columns([1, 1.4])
with col3:
    seg = customers["credit_band"].value_counts().sort_index()
    labels = [s.split(". ", 1)[1] for s in seg.index.astype(str)]
    fig3 = px.pie(values=seg.values, names=labels, title="Customer Segment Mix (by Credit Band)",
                 color_discrete_sequence=[BERRY, PURPLE, TEAL, NAVY, "#8FBF8F"])
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown("#### Jump to detail")
    st.markdown(
        """
This dashboard deliberately has **no drill-through** — it's meant to *link out* to the other
3 dashboards for investigation, not replace them:
- 🏦 **Branch Operations** — branch-level deposit & account detail
- 💳 **Credit Risk** — loan-level model scores & customer drill-down
- 🛡️ **Fraud & Compliance** — transaction-level investigation queue

Use the sidebar navigation to open any of them.
"""
    )
