import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Branch Operations — PRP Project 2", page_icon="🏦", layout="wide")

from utils.style import inject_css, page_header, callout, NAVY, TEAL, PURPLE, BERRY
from utils.data_loader import load_accounts, load_branches, load_cards

inject_css()

page_header(
    "Dashboard 1 — Branch Operations",
    "Stakeholder: Branch Managers · Real-time view of account portfolio & deposit health per branch",
    tag="DASHBOARD 1 OF 4",
)

accounts = load_accounts()
branches = load_branches()
cards = load_cards()

# ---------------------------------------------------------------- Slicers
st.sidebar.markdown("### Filters")
branch_options = ["All"] + sorted(branches["branch_name"].unique().tolist())
sel_branch = st.sidebar.selectbox("Branch", branch_options)
sel_type = st.sidebar.multiselect("Account Type", sorted(accounts["account_type"].unique()),
                                   default=sorted(accounts["account_type"].unique()))
min_d, max_d = accounts["open_date"].min(), accounts["open_date"].max()
date_range = st.sidebar.slider("Open Date Range", min_value=min_d.to_pydatetime(),
                                max_value=max_d.to_pydatetime(),
                                value=(min_d.to_pydatetime(), max_d.to_pydatetime()))

acc = accounts.merge(branches[["branch_id", "branch_name", "manager_name"]], on="branch_id", how="left")
acc = acc[acc["account_type"].isin(sel_type)]
acc = acc[(acc["open_date"] >= date_range[0]) & (acc["open_date"] <= date_range[1])]
if sel_branch != "All":
    acc = acc[acc["branch_name"] == sel_branch]

# ---------------------------------------------------------------- KPIs
total_aum = acc["balance_usd"].sum()
account_count = len(acc)
avg_balance = acc["balance_usd"].mean() if account_count else 0

today = pd.Timestamp.now().normalize()
cards_expiring = cards[(cards["expiration_date"] >= today) & (cards["expiration_date"] <= today + pd.Timedelta(days=30))]
cards_expiring = cards_expiring.merge(acc[["account_id"]], on="account_id", how="inner") if account_count else cards_expiring.iloc[0:0]

latest_open = acc["open_date"].max() if account_count else accounts["open_date"].max()
new_accounts = acc[(acc["open_date"] > latest_open - pd.Timedelta(days=30)) & (acc["open_date"] <= latest_open)]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total AUM", f"${total_aum/1e9:.2f}B" if total_aum >= 1e9 else f"${total_aum/1e6:.1f}M")
c2.metric("Account Count", f"{account_count:,}")
c3.metric("Avg Balance", f"${avg_balance:,.0f}")
c4.metric("Cards Expiring (30 Days)", f"{len(cards_expiring):,}")
c5.metric("New Accounts (Last 30 Days)", f"{len(new_accounts):,}")

st.divider()

col1, col2 = st.columns(2)
with col1:
    branch_aum = acc.groupby("branch_name")["balance_usd"].sum().sort_values(ascending=False).head(20)
    fig = px.bar(x=branch_aum.values, y=branch_aum.index, orientation="h",
                 labels={"x": "Total AUM (USD)", "y": ""}, title="AUM by Branch (Top 20)",
                 color_discrete_sequence=[NAVY])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    deposit_trend = acc.groupby("open_month")["balance_usd"].sum().reset_index()
    fig2 = px.line(deposit_trend, x="open_month", y="balance_usd", title="Deposit Trend (by Account Open Month)",
                   color_discrete_sequence=[TEAL])
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "This dataset has no periodic balance snapshot — a balance is each account's current "
        "balance, not a running ledger. 'Trend' here means: for accounts opened in a given month, "
        "what's their combined balance today."
    )

col3, col4 = st.columns([1, 1.6])
with col3:
    mix = acc["account_type"].value_counts()
    fig3 = px.pie(values=mix.values, names=mix.index, title="Account Type Mix",
                 color_discrete_sequence=[NAVY, TEAL, PURPLE])
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown("#### Branches Table")
    branch_table = acc.groupby(["branch_name", "manager_name"]).agg(
        account_count=("account_id", "count"), total_aum=("balance_usd", "sum"), avg_balance=("balance_usd", "mean")
    ).sort_values("total_aum", ascending=False).reset_index()
    st.dataframe(
        branch_table.style.format({"total_aum": "${:,.0f}", "avg_balance": "${:,.0f}"}),
        use_container_width=True, height=380,
    )

st.divider()
st.markdown("#### Branch Detail (drill-through)")
drill_branch = st.selectbox("Select a branch to drill into its individual accounts", branch_table["branch_name"] if len(branch_table) else [])
if drill_branch:
    detail = acc[acc["branch_name"] == drill_branch][["account_id", "account_type", "balance_usd", "open_date"]]
    st.dataframe(detail.sort_values("balance_usd", ascending=False), use_container_width=True, height=300)
    st.caption(f"{len(detail):,} accounts at {drill_branch}.")

callout(
    "Branch assignment (branch_id) is synthetically engineered for this dataset; branch city/"
    "country fields are borrowed from a mismatched source table and carry no real correspondence "
    "to the actual branch. See Data Quality & Limitations.",
)
