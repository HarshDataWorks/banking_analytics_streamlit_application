import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Diagnostic Analysis — PRP Project 2", page_icon="🔍", layout="wide")

from utils.style import inject_css, page_header, callout, NAVY, TEAL, PURPLE, BERRY, LOW_COLOR, MEDIUM_COLOR, HIGH_COLOR
from utils.data_loader import load_customers, load_accounts, load_loans
from utils.fraud import compute_fraud_scores, fraud_rejected_rules_evidence

inject_css()

page_header(
    "Diagnostic Analysis",
    "Why the numbers look the way they do — percentiles, ratios, group averages, and plain "
    "correlation only. No z-scores, no p-values, no ANOVA, no regression.",
    tag="PHASE 2 OF 3",
)

st.markdown(
    """
Descriptive analysis told us **what** the numbers are. This page digs into **why** — for every
major descriptive finding that looked surprising or flat, we investigate the root cause using
tools already used earlier in this project: percentiles and correlation were both used in the SQL
KPI layer, just pointed at a "why" question instead of a "what" question.
"""
)

tab1, tab2 = st.tabs(["🚩 Fraud Risk Diagnostics", "🔎 Root-Cause Diagnostics"])

# ============================================================ TAB 1: FRAUD
with tab1:
    st.markdown("#### A simple, count-based flagging system")
    st.markdown(
        """
Four simple rules, each a yes/no flag on every transaction. A transaction's
`derived_fraud_score` is just **how many of the 4 rules it breaks** (0 to 4) — no weights, no
formulas, just a count. This mirrors how a human fraud analyst triages a queue: "this one tripped
2 different red flags, look at it before the one that only tripped 1."
"""
    )

    with st.spinner("Computing fraud scores across all 1,000,000 transactions (cached after first run)..."):
        tx = compute_fraud_scores()

    tier_counts = tx["fraud_tier"].value_counts().reindex(["Low", "Medium", "High"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Low (0 rules)", f"{tier_counts['Low']:,}", f"{100*tier_counts['Low']/len(tx):.2f}%")
    c2.metric("Medium (1 rule)", f"{tier_counts['Medium']:,}", f"{100*tier_counts['Medium']/len(tx):.2f}%")
    c3.metric("High (2+ rules)", f"{tier_counts['High']:,}", f"{100*tier_counts['High']/len(tx):.2f}%")
    c4.metric("Investigation queue (Medium+High)", f"{tier_counts['Medium']+tier_counts['High']:,}")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        fig = px.bar(x=["Low", "Medium", "High"], y=tier_counts.values, log_y=True,
                     labels={"x": "Fraud Tier", "y": "Transaction Count (log scale)"},
                     title="Fraud Tier Counts (1,000,000 transactions)",
                     color=["Low", "Medium", "High"],
                     color_discrete_map={"Low": LOW_COLOR, "Medium": MEDIUM_COLOR, "High": HIGH_COLOR})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rule_counts = {
            "Amount anomaly\n(>2x own avg)": int(tx["flag_amount_anomaly"].sum()),
            "Velocity anomaly\n(3+ in 10min)": int(tx["flag_velocity_anomaly"].sum()),
            "Merchant diversity\n(3+ in 30 days)": int(tx["flag_merchant_diversity_anomaly"].sum()),
            "Geo improbability\n(diff. city, <60min)": int(tx["flag_geo_improbability"].sum()),
        }
        fig2 = px.bar(x=list(rule_counts.keys()), y=list(rule_counts.values()),
                      labels={"x": "", "y": "Transactions Flagged"},
                      title="Flagged Transactions by Rule Broken",
                      color_discrete_sequence=[NAVY])
        st.plotly_chart(fig2, use_container_width=True)

    callout(
        f"<b>95.36% Low / 4.60% Medium / 0.04% High</b> — a ~46,400-transaction Medium+High queue "
        f"for the Compliance dashboard. Only {int((tx['derived_fraud_score']>=2).sum()):,} "
        f"transactions break 2+ rules at once.",
        kind="finding",
    )

    st.markdown("#### Why per-account, not one fixed dollar amount for everyone")
    st.markdown(
        """
A Business account that normally spends $8,000 a transaction isn't doing anything unusual with a
$9,000 purchase. An account that normally spends $50 making a $9,000 purchase absolutely is.
Comparing each transaction to **that account's own average**, not a single number for the whole
bank, is the only version of "big" that actually means something here.
"""
    )

    st.markdown("#### Rules tried and deliberately **not** used — and why")
    ev = fraud_rejected_rules_evidence()

    col3, col4 = st.columns(2)
    with col3:
        hc = ev["hour_counts"]
        fig3 = px.bar(x=hc.index, y=hc.values, labels={"x": "Hour of Day", "y": "Transaction Count"},
                      title="Transaction Count by Hour — why the 'odd hour' rule fails",
                      color_discrete_sequence=[PURPLE])
        st.plotly_chart(fig3, use_container_width=True)
        callout(
            f"Every hour has almost exactly the same number of transactions "
            f"({hc.min():,} to {hc.max():,} — only ~2% spread, just noise). A midnight-5am rule "
            f"would flag {ev['oddhour_pct']:.1f}% of ALL transactions simply because that window "
            f"IS 5 of 24 hours (5/24 = 20.8%) — not because it catches anything real. "
            f"<b>Decision: not used.</b>",
            kind="finding",
        )

    with col4:
        st.markdown(
            f"""
**Velocity rule (3+ transactions within 10 minutes): catches {int(tx['flag_velocity_anomaly'].sum())}
transactions.**

The typical (median) gap between one account's transactions is about
**{ev['median_gap_days']:,.0f} days** (~{ev['median_gap_minutes']:,.0f} minutes). With an average
of only ~13.3 transactions per account spread across 7 years, this dataset's transactions per
account are far too spread out in time for a "burst of purchases" to ever occur for anyone.

This is a genuine limitation of the **data**, not a mistake in the rule — a real burst-buying
pattern (a stolen card used rapidly) is exactly what this rule is built to catch, and this
dataset simply never produces that clustering.
"""
        )
        callout(
            "Both rejected/near-zero rules trace back to the same root cause already found in the "
            "descriptive phase: transaction timing and frequency were generated close to uniformly, "
            "with no real daily rhythm or clustering built in.",
        )

# ============================================================ TAB 2: ROOT CAUSE
with tab2:
    customers = load_customers()
    accounts = load_accounts()
    loans = load_loans()

    st.markdown("#### 1. Why doesn't credit score predict loan terms?")
    lc = loans.merge(customers[["customer_id", "credit_score"]], on="customer_id")
    c1, c2, c3 = st.columns(3)
    c1.metric("credit_score ↔ loan_amount", f"{lc['credit_score'].corr(lc['loan_amount']):+.4f}")
    c2.metric("credit_score ↔ interest_rate", f"{lc['credit_score'].corr(lc['interest_rate']):+.4f}")
    c3.metric("loan_amount ↔ interest_rate", f"{lc['loan_amount'].corr(lc['interest_rate']):+.4f}")

    fig = px.scatter(lc.sample(min(20000, len(lc)), random_state=1), x="credit_score", y="interest_rate",
                     opacity=0.2, title="Credit Score vs. Interest Rate — a shapeless cloud is what 'no relationship' looks like",
                     color_discrete_sequence=[NAVY])
    st.plotly_chart(fig, use_container_width=True)
    callout(
        "All three correlations are effectively 0 — as close to 'no relationship at all' as real "
        "numbers get. This is the direct answer to WHY the descriptive-phase band averages looked "
        "so flat: not a coincidence of averaging — these variables are genuinely unrelated at the "
        "individual-loan level, most likely because this synthetic data was generated without real "
        "underwriting logic connecting risk to pricing or loan size.",
        kind="finding",
    )

    st.divider()
    st.markdown("#### 2. Why do some branches carry more money than others?")
    branch_stats = accounts.groupby("branch_id").agg(
        account_count=("account_id", "count"), total_aum=("balance_usd", "sum")
    ).reset_index()
    branch_stats["aum_per_account"] = branch_stats["total_aum"] / branch_stats["account_count"]

    fig2 = px.scatter(branch_stats, x="account_count", y="aum_per_account",
                      title="If branch size mattered beyond headcount, this would trend up or down. It doesn't.",
                      color_discrete_sequence=[TEAL])
    st.plotly_chart(fig2, use_container_width=True)
    callout(
        f"AUM per account is almost identical everywhere — "
        f"${branch_stats['aum_per_account'].min():,.0f} to ${branch_stats['aum_per_account'].max():,.0f} "
        f"across all 500 branches, centered around ${branch_stats['aum_per_account'].mean():,.0f}. "
        f"A branch with 10x the accounts of another simply has roughly 10x the money — no more, no "
        f"less. Bigger branches have more accounts, not 'better' accounts.",
        kind="finding",
    )

    st.markdown("##### Is the fraud queue clustered at certain branches, or spread evenly?")
    with st.spinner("Joining fraud scores to branches..."):
        tx2 = compute_fraud_scores()
        tx_branch = tx2.merge(accounts[["account_id", "branch_id"]], on="account_id", how="left")
        branch_fraud = tx_branch.groupby("branch_id").agg(
            txn_count=("transaction_id", "count"),
            flagged_count=("fraud_tier", lambda s: (s != "Low").sum()),
        )
        branch_fraud["pct_flagged"] = 100 * branch_fraud["flagged_count"] / branch_fraud["txn_count"]

    fig3 = px.histogram(branch_fraud, x="pct_flagged", nbins=25,
                        title="% of Each Branch's Transactions Flagged (Medium or High)",
                        color_discrete_sequence=[BERRY])
    fig3.add_vline(x=branch_fraud["pct_flagged"].mean(), line_dash="dash", line_color="black")
    st.plotly_chart(fig3, use_container_width=True)
    callout(
        f"Every branch sits in a narrow band, roughly {branch_fraud['pct_flagged'].min():.1f}% to "
        f"{branch_fraud['pct_flagged'].max():.1f}% flagged, centered around "
        f"{branch_fraud['pct_flagged'].mean():.1f}%. No branch is a dramatic standout — makes sense "
        f"since branch_id was an engineered, effectively random assignment, unrelated to an "
        f"account's own transaction history (which is all the fraud rules look at).",
    )

    st.divider()
    st.markdown("#### 3. Do Business accounts transact differently? Does tenure drive balance?")
    col5, col6 = st.columns(2)
    with col5:
        tx_acc = tx2.merge(accounts[["account_id", "account_type"]], on="account_id", how="left")
        behavior = tx_acc.groupby("account_type")["amount_usd"].agg(avg_amount="mean", txn_count="count")
        st.dataframe(behavior.round(2), use_container_width=True)
        callout(
            "All three account types average almost exactly $5,000 per transaction. account_type "
            "simply doesn't drive transaction behavior in this dataset — the same way it doesn't "
            "drive balance either.",
        )

    with col6:
        today = pd.Timestamp.now().normalize()
        cust_acc = accounts.merge(customers[["customer_id", "created_at"]], on="customer_id")
        cust_acc["tenure_years"] = (today - cust_acc["created_at"]).dt.days / 365.25
        per_cust = cust_acc.groupby("customer_id").agg(
            tenure_years=("tenure_years", "first"), avg_balance=("balance_usd", "mean")
        )
        corr_tb = per_cust["tenure_years"].corr(per_cust["avg_balance"])
        fig4 = px.scatter(per_cust.sample(min(10000, len(per_cust)), random_state=1),
                          x="tenure_years", y="avg_balance", opacity=0.15,
                          title=f"Tenure vs. Avg Balance (r = {corr_tb:+.4f})",
                          color_discrete_sequence=[NAVY])
        st.plotly_chart(fig4, use_container_width=True)
        callout(
            f"Correlation is {corr_tb:+.4f} — essentially zero. Being a customer for 7 years "
            f"doesn't correlate with holding more money than someone who joined last month.",
            kind="finding",
        )

    st.markdown(
        """
**The single biggest, most re-usable takeaway from this whole diagnostic phase:** variables that
WOULD be connected at a real bank — credit score & loan terms, account type & behavior, tenure &
balance, branch identity & AUM or fraud risk — are all independent of each other here. See the
Home page for the full "Independence Story" list.
"""
    )
