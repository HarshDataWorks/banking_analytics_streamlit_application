import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Fraud & Compliance — PRP Project 2", page_icon="🛡️", layout="wide")

from utils.style import inject_css, page_header, callout, NAVY, TEAL, PURPLE, BERRY, LOW_COLOR, MEDIUM_COLOR, HIGH_COLOR
from utils.fraud import compute_fraud_scores

inject_css()

page_header(
    "Dashboard 3 — Fraud & Compliance",
    "Stakeholder: Compliance/Fraud Team · Flagged transactions in real time, prioritized for investigation",
    tag="DASHBOARD 3 OF 4",
)

with st.spinner("Loading fraud scores (cached after first run)..."):
    tx = compute_fraud_scores()

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------- Slicers
st.sidebar.markdown("### Filters")
tier_opts = ["Low", "Medium", "High"]
sel_tier = st.sidebar.multiselect("Risk Tier", tier_opts, default=tier_opts)
min_d, max_d = tx["transaction_date"].min(), tx["transaction_date"].max()
date_range = st.sidebar.slider("Date Range", min_value=min_d.to_pydatetime(), max_value=max_d.to_pydatetime(),
                                value=(min_d.to_pydatetime(), max_d.to_pydatetime()))
merchant_search = st.sidebar.text_input("Merchant ID contains", "")
account_search = st.sidebar.text_input("Account ID contains", "")

f = tx[tx["fraud_tier"].isin(sel_tier) & tx["transaction_date"].between(date_range[0], date_range[1])]
if merchant_search:
    f = f[f["merchant_id"].str.contains(merchant_search, case=False, na=False)]
if account_search:
    f = f[f["account_id"].str.contains(account_search, case=False, na=False)]

# ---------------------------------------------------------------- KPIs
flagged = f[f["fraud_tier"] != "Low"]
total_flagged = len(flagged)
flagged_volume = flagged["amount_usd"].sum()
pct_volume = 100 * flagged_volume / f["amount_usd"].sum() if len(f) and f["amount_usd"].sum() else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total Flagged Transactions", f"{total_flagged:,}")
c2.metric("Flagged $ Volume", f"${flagged_volume/1e6:.1f}M")
c3.metric("% of Volume Flagged", f"{pct_volume:.1f}%")

if flagged.empty:
    st.info("No flagged (Medium/High) transactions match the current filters — widen the filters in the sidebar to see the charts below.")
    st.stop()

st.markdown("##### Top flagged accounts / merchants")
col_a, col_b = st.columns(2)
with col_a:
    top_acc = flagged["account_id"].value_counts().head(10).reset_index()
    top_acc.columns = ["account_id", "flagged_count"]
    st.dataframe(top_acc, use_container_width=True, height=250)
with col_b:
    top_merch = flagged["merchant_id"].value_counts().head(10).reset_index()
    top_merch.columns = ["merchant_id", "flagged_count"]
    st.dataframe(top_merch, use_container_width=True, height=250)

st.divider()

col1, col2 = st.columns([1.3, 1])
with col1:
    heat = flagged.groupby(["day_of_week", "hour"]).size().reset_index(name="flagged_count")
    heat_pivot = heat.pivot(index="day_of_week", columns="hour", values="flagged_count").reindex(dow_order).fillna(0)
    fig = px.imshow(heat_pivot, aspect="auto", color_continuous_scale="YlOrRd",
                    title="Fraud Flags Heatmap — Day of Week × Hour", labels={"color": "Flagged Count"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Intentionally close to flat — none of the 4 fraud rules use time-of-day as a signal, "
        "since the diagnostic phase found no real time-based pattern in this data to build one from."
    )

with col2:
    rule_counts = {
        "Amount": int(flagged["flag_amount_anomaly"].sum()),
        "Velocity": int(flagged["flag_velocity_anomaly"].sum()),
        "Merchant Div.": int(flagged["flag_merchant_diversity_anomaly"].sum()),
        "Geo": int(flagged["flag_geo_improbability"].sum()),
    }
    fig2 = px.bar(x=list(rule_counts.keys()), y=list(rule_counts.values()),
                 title="Flagged by Rule Broken", color_discrete_sequence=[BERRY])
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("This is what actually drives the queue's composition — pair with the heatmap above.")

col3, col4 = st.columns(2)
with col3:
    tier_counts = f["fraud_tier"].value_counts().reindex(tier_opts).fillna(0)
    fig3 = px.bar(x=tier_counts.index, y=tier_counts.values, log_y=True,
                  title="Flagged Transactions by Risk Tier (log scale)",
                  color=tier_counts.index,
                  color_discrete_map={"Low": LOW_COLOR, "Medium": MEDIUM_COLOR, "High": HIGH_COLOR})
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    trend = flagged.groupby("transaction_month").size().reset_index(name="flagged_count")
    fig4 = px.line(trend, x="transaction_month", y="flagged_count", title="Flagged Volume Trend",
                   color_discrete_sequence=[NAVY])
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.markdown("#### Investigation Table")
inv = flagged.sort_values("derived_fraud_score", ascending=False)[
    ["transaction_id", "account_id", "merchant_id", "amount_usd", "transaction_date", "derived_fraud_score", "fraud_tier"]
]
st.dataframe(inv.head(300).style.format({"amount_usd": "${:,.2f}"}), use_container_width=True, height=350)

st.markdown("#### Account Investigation (drill-through)")
drill_acc = st.text_input("Enter an account_id to see its ENTIRE transaction history (flagged or not)", "")
if drill_acc:
    hist = tx[tx["account_id"] == drill_acc][
        ["transaction_id", "merchant_id", "amount_usd", "transaction_date", "derived_fraud_score", "fraud_tier"]
    ].sort_values("transaction_date")
    st.dataframe(hist.style.format({"amount_usd": "${:,.2f}"}), use_container_width=True, height=300)
    st.caption(f"{len(hist):,} total transactions for account {drill_acc}.")

callout(
    "A compliance analyst investigating one suspicious transaction almost always wants to see that "
    "account's whole history next — is this a one-off, or part of a pattern? The drill-through "
    "above shows the full account history, flagged or not, for exactly that reason.",
)
