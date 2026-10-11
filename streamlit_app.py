"""
PRP Project 2 — Synthetic Banking Dataset Analytics Platform
Home / Executive Summary page.

This is the Streamlit deployment of the full project: SQL data quality +
cleaning, descriptive + diagnostic + predictive analysis, and all 4 Power BI
dashboards, re-built here as one interactive, live-computed web app so
every number on every page is calculated straight from the same 7 raw CSVs
the rest of the project used — nothing here is a static screenshot.
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PRP Project 2 — Banking Analytics Platform",
    page_icon="🏦",
    layout="wide",
)

from utils.style import inject_css, page_header, callout, NAVY, TEAL, PURPLE, BERRY
from utils.data_loader import load_customers, load_accounts, load_loans, load_branches, totals

inject_css()

page_header(
    "PRP Project 2 — Banking Analytics Platform",
    "Synthetic Banking Dataset · 7 tables · 1,260,500 rows · Excel → SQL (Snowflake) → "
    "Python (Databricks) → Power BI → Streamlit",
    tag="DATA ANALYST INTERNSHIP — FINAL DELIVERABLE",
)

st.markdown(
    """
Use the sidebar to move through the project in the same order it was built: **Descriptive →
Diagnostic → Predictive**, then the four dashboard-equivalent pages (Branch Operations, Credit
Risk, Fraud & Compliance, Executive Overview), and finally **Data Quality & Limitations** — the
page that discloses everything synthetic, engineered, or constructed in this dataset, since every
number elsewhere in this app depends on understanding those limitations honestly.
"""
)

t = totals()
customers = load_customers()
accounts = load_accounts()
loans = load_loans()
branches = load_branches()

total_aum = accounts["balance_usd"].sum()
total_loan_book = loans["loan_amount"].sum()

st.markdown("### Headline numbers")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers", f"{t['customers']:,}")
c2.metric("Accounts", f"{t['accounts']:,}")
c3.metric("Total AUM", f"${total_aum/1e9:,.2f}B")
c4.metric("Loans", f"{t['loans']:,}")
c5.metric("Total Loan Book", f"${total_loan_book/1e9:,.2f}B")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Cards", f"{t['cards']:,}")
c7.metric("Merchants", f"{t['merchants']:,}")
c8.metric("Branches", f"{t['branches']:,}")
c9.metric("Transactions", f"{t['transactions']:,}")
c10.metric("Total Rows (all tables)", f"{t['total_rows']:,}")

st.divider()

left, right = st.columns([1.3, 1])

with left:
    st.markdown("### The business problem")
    st.markdown(
        """
An anonymous retail/commercial bank has no systematic, data-driven view of two things that
directly affect its bottom line: **transaction-level fraud risk** (no ground-truth fraud label
exists) and **loan default risk** (no predictive way to flag at-risk loans before default). This
project builds a rule-based fraud-flagging system, one flagship logistic-regression default-risk
model, and real-time dashboards for four stakeholder groups — branch managers, the credit risk
team, compliance/fraud, and executive leadership.
"""
    )

    st.markdown("### The single biggest finding: \"The Independence Story\"")
    callout(
        """
Eight separate variable pairs that <b>should</b> be related at a real bank turn out to be
statistically independent in this dataset — this single pattern explains why several KPIs look
unusually flat, and why the flagship model's target had to be <b>constructed</b> rather than
learned naturally:
<ul>
<li>credit_score ↔ loan_amount (r ≈ -0.0004)</li>
<li>credit_score ↔ interest_rate (r ≈ +0.0041)</li>
<li>loan_amount ↔ interest_rate (r ≈ -0.0045)</li>
<li>account_type ↔ balance_usd (all 3 types average ~$100,150-$100,240)</li>
<li>account_type ↔ transaction behavior (all 3 types ~$5,000 avg/txn)</li>
<li>branch identity ↔ AUM-per-account ($82,600-$119,000, no size trend)</li>
<li>branch identity ↔ fraud-flag rate (every branch sits 2.9%-7.8%)</li>
<li>customer tenure ↔ average balance (r ≈ +0.0030)</li>
</ul>
Most likely explanation: this synthetic dataset generated these columns independently of each
other, without the real underwriting/behavioral logic that connects them at an actual bank.
""",
        kind="finding",
    )

with right:
    st.markdown("### Credit score distribution")
    band_counts = customers["credit_band"].value_counts().sort_index()
    band_counts.index = [s.split(". ", 1)[1] for s in band_counts.index.astype(str)]
    st.bar_chart(band_counts, color=NAVY)
    st.caption(
        "50.8% of customers fall in 'Poor' alone — the opposite skew of a real FICO population. "
        "See Data Quality & Limitations for the full disclosure."
    )

st.divider()

st.markdown("### What management should do next")
r1, r2 = st.columns(2)
with r1:
    st.markdown("""
**On fraud & compliance:**
1. Stand up the ~46,400-transaction Medium/High investigation queue as a live compliance
   workflow — it's small enough (4.64% of volume) to action, not just monitor.
2. Treat the amount-anomaly rule (35,666 hits) as the primary signal; de-prioritize time-of-day
   and velocity-based detection investment until richer behavioral data exists.
3. If a real deployment follows this pilot, prioritize acquiring genuine geolocation and
   online/in-person channel data — both are missing here and limit the geo rule to a weak proxy.
""")
with r2:
    st.markdown("""
**On credit risk:**
1. Use the model's rank-ordering (not just the 0.21 cutoff) operationally — the top decile is
   ~88% genuinely high-risk, which supports a tiered review process, not a single pass/fail gate.
2. Because `credit_score`, `loan_amount`, and `interest_rate` are generated independently here,
   this model's coefficients should be re-validated the moment real underwriting data is available
   — don't carry these exact odds ratios into production untested.
3. Invest in closing the data gaps disclosed on the Limitations page (timeline-logic violations,
   no account/card/loan status field) before scaling any of this beyond a pilot.
""")

st.caption(
    "Tech stack: Excel · SQL (Snowflake) · Python (Databricks) · Power BI · Streamlit · GitHub. "
    "Predictive modeling restricted to Linear/Logistic Regression per the project's own scope rules."
)
