import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Quality & Limitations — PRP Project 2", page_icon="⚠️", layout="wide")

from utils.style import inject_css, page_header, callout, NAVY, TEAL, BERRY
from utils.data_loader import load_customers, load_accounts, load_loans, load_transactions, load_branches, load_merchants, load_cards

inject_css()

page_header(
    "Data Quality & Limitations",
    "Every synthetic, engineered, or constructed element in this dataset, disclosed in one place — "
    "every number elsewhere in this app depends on understanding these honestly.",
    tag="DISCLOSURE",
)

customers = load_customers()
accounts = load_accounts()
loans = load_loans()
branches = load_branches()
merchants = load_merchants()
cards = load_cards()

st.markdown("### 1. Timeline logic violations — the single biggest data-quality finding")
col1, col2, col3 = st.columns(3)
with col1:
    bad_acc = int(accounts["date_logic_flag"].sum())
    col1.metric("Accounts affected", f"{bad_acc:,}", f"{100*bad_acc/len(accounts):.1f}% of {len(accounts):,}")
with col2:
    bad_loans = int(loans["date_logic_flag"].sum())
    col2.metric("Loans affected", f"{bad_loans:,}", f"{100*bad_loans/len(loans):.1f}% of {len(loans):,}")
with col3:
    with st.spinner("Loading transactions to check timeline logic (cached after first run)..."):
        tx = load_transactions()
    bad_tx = int(tx["date_logic_flag"].sum())
    col3.metric("Transactions affected", f"{bad_tx:,}", f"{100*bad_tx/len(tx):.1f}% of {len(tx):,}")

callout(
    "Roughly half of accounts, loans, and transactions carry a date earlier than their parent "
    "record's own creation date — e.g. an account opened before its owning customer existed. "
    "Impossible in a real bank. <b>Cleaning philosophy: FLAG, never DELETE.</b> There is no ground "
    "truth to correct these dates TO, so any 'fix' would mean fabricating data — worse for "
    "audit-defensibility than flagging. Every KPI in this app runs on the FULL table by default, "
    "exactly like the SQL/Python phases did.",
    kind="warn",
)

st.divider()
st.markdown("### 2. Engineered / synthetic fields — never present these as authentic")

st.markdown(
    """
| Field | Status | Detail |
|---|---|---|
| `accounts.branch_id` | **Engineered** | Did not exist in the source data (branches was a fully orphaned reference table). Assigned via a seeded (seed=42) lognormal-weighted random draw (mean=0, sigma=0.6) to mimic realistic branch-size variance — not a real operational assignment. |
| `branches.city` / `branches.country` | **Synthetic / borrowed** | Positionally concatenated from a *different* branch_id set (a separate raw export). There is no real ID correspondence between these values and that row's actual branch_id. `country` is 100% "USA" with zero variation — a country filter/slicer would be functionally meaningless. |
| `derived_fraud_score` / `fraud_tier` | **Rule-based, never modeled** | No ground-truth fraud label exists anywhere in this data — there is nothing to train or validate a model against. The score is a plain count (0-4) of simple, disclosed rules, diagnostic-only, forever. |
| `derived_default_risk_flag` | **Constructed target** | No "did this loan actually default" field exists. Built from a simple, disclosed rule (Poor credit_score AND top-25% loan_amount or interest_rate) — never a real observed outcome. |
"""
)

col4, col5 = st.columns(2)
with col4:
    country_counts = branches["country"].value_counts()
    st.metric("Distinct branch countries", f"{len(country_counts)}", "100% USA — zero variation")

with col5:
    sizes = accounts.groupby("branch_id").size()
    st.metric("Branch size range (accounts/branch)", f"{sizes.min()}–{sizes.max():,}",
              f"mean {sizes.mean():.0f}, median {sizes.median():.0f}")

st.divider()
st.markdown("### 3. Unrealistically clean & unrealistically uniform data")

col6, col7, col8 = st.columns(3)
with col6:
    nulls = sum(df.isnull().sum().sum() for df in [customers, accounts, loans, branches, merchants, cards])
    st.metric("Total NULLs across all 6 small tables", f"{int(nulls)}")
with col7:
    dup_emails = customers["email"].duplicated(keep=False).sum()
    st.metric("Duplicate customer emails", f"{dup_emails:,}",
              f"{customers['email'].value_counts().gt(1).sum():,} emails reused")
with col8:
    n_unique_cities = customers["city"].nunique()
    st.metric("Customer city uniqueness", f"{100*n_unique_cities/len(customers):.1f}%",
              f"{n_unique_cities:,} distinct / {len(customers):,} customers")

callout(
    "Zero nulls and zero duplicate rows across every original table — real banking data is never "
    "this clean. Customer cities are near-unique (largest city holds only ~51 of 50,000 "
    "customers) — an unrealistic geographic distribution that is why the Executive Overview "
    "dashboard deliberately uses a bar chart instead of a map.",
)

col9, col10 = st.columns(2)
with col9:
    zero_acc_customers = len(customers) - accounts["customer_id"].nunique()
    st.metric("Customers with zero accounts", f"{zero_acc_customers:,}",
              f"{100*zero_acc_customers/len(customers):.1f}%")
with col10:
    today = pd.Timestamp.now().normalize()
    expired = (cards["expiration_date"] < today).sum()
    st.metric("Cards already expired (as of today)", f"{expired:,}",
              f"~{100*expired/len(cards):.1f}% — shifts daily, never quote the exact digit")

st.divider()
st.markdown("### 4. Referential integrity — what actually passed")
st.markdown(
    """
All 7 primary keys are unique. All 6 foreign-key relationships resolve with **zero orphaned
rows**: `accounts→customers`, `accounts→branches`, `cards→accounts`, `loans→customers`,
`transactions→accounts`, `transactions→merchants`. No negative/zero values in `balance_usd`,
`loan_amount`, or `amount_usd`. No `credit_score` outside the valid 300-850 range.
"""
)

st.divider()
st.markdown("### 5. Assumptions made about this data")
st.markdown(
    """
1. A single, unnamed retail/commercial bank.
2. "Business" account_type means small-business, not enterprise.
3. All monetary values are USD.
4. All branches operate in a single country (confirmed: 100% "USA").
5. Every loan row is currently active/open — no status field exists.
6. Every card row is currently issued/active — no status field exists.
7. Every transaction row is completed/settled — no status field exists.
8. Fraud/default-risk thresholds (the 4 fraud rules, the 580 credit-score cutoff, the P75 loan/rate
   thresholds) are analyst-chosen defaults, not client-specified business rules.
"""
)

st.markdown("### 6. Clarifications that would be raised with a real client")
st.markdown(
    """
1. Is any ground-truth fraud-labeled data available?
2. Is a real loan repayment/default status field available elsewhere?
3. What risk appetite should thresholds reflect — recall or precision?
4. Should Business-type accounts be a distinct reporting segment?
5. Is authoritative branch city/country data available?
6. What time zone do transaction timestamps represent?
7. Should closed/inactive accounts/cards/loans (if such a status existed) be excluded from
   portfolio KPIs?
"""
)

st.divider()
callout(
    "<b>Naming discipline, carried through every page of this app:</b> every derived/synthetic "
    "field is named with a <code>derived_</code> prefix or clearly labeled 'engineered'/'synthetic' "
    "— never presented as if it were ground truth observed at a real bank.",
    kind="finding",
)
