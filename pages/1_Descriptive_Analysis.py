import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Descriptive Analysis — PRP Project 2", page_icon="📊", layout="wide")

from utils.style import inject_css, page_header, callout, NAVY, TEAL, PURPLE, BERRY, PLOTLY_SEQUENCE
from utils.data_loader import load_customers, load_accounts, load_cards, load_loans, load_branches, load_merchants

inject_css()

page_header(
    "Descriptive Analysis",
    "What the numbers are — Areas 1-6, mirroring 05_kpi_definitions.sql and 06_descriptive_eda.py",
    tag="PHASE 1 OF 3",
)

customers = load_customers()
accounts = load_accounts()
cards = load_cards()
loans = load_loans()
branches = load_branches()
merchants = load_merchants()

area = st.radio(
    "Jump to an area",
    ["1. Customer Base", "2. Account Portfolio", "3. Card Portfolio",
     "4. Loan Book", "5. Branch Network"],
    horizontal=True,
)

st.divider()

# ============================================================ AREA 1
if area.startswith("1"):
    st.subheader("Area 1 — Customer Base")

    col1, col2 = st.columns(2)
    with col1:
        band_counts = customers["credit_band"].value_counts().sort_index()
        band_pct = (band_counts / band_counts.sum() * 100).round(1)
        labels = [s.split(". ", 1)[1] for s in band_counts.index.astype(str)]
        fig = px.bar(x=labels, y=band_counts.values, text=[f"{p}%" for p in band_pct.values],
                     labels={"x": "", "y": "Customers"}, title="Credit Score Distribution by Band",
                     color_discrete_sequence=[NAVY])
        st.plotly_chart(fig, use_container_width=True)
        callout(
            "50.8% of customers fall in the 'Poor' band alone — the <b>opposite</b> skew of a "
            "real FICO population, which typically clusters Good/Very Good. Likely generated from "
            "something close to a uniform distribution across 300-850, not a realistic population.",
            kind="finding",
        )

    with col2:
        cohort_counts = customers["tenure_cohort"].value_counts().sort_index()
        labels2 = [s.split(". ", 1)[1] for s in cohort_counts.index.astype(str)]
        fig2 = px.bar(x=labels2, y=cohort_counts.values, labels={"x": "", "y": "Customers"},
                      title="Customer Tenure Cohorts", color_discrete_sequence=[TEAL])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        top_cities = customers["city"].value_counts().head(20).sort_values()
        fig3 = px.bar(x=top_cities.values, y=top_cities.index, orientation="h",
                      labels={"x": "Customers", "y": ""}, title="Top 20 Customer Cities",
                      color_discrete_sequence=[PURPLE])
        st.plotly_chart(fig3, use_container_width=True)
        n_unique = customers["city"].nunique()
        callout(
            f"{n_unique:,} distinct cities across {len(customers):,} customers "
            f"({100*n_unique/len(customers):.1f}% unique) — the single largest city holds only "
            f"~{top_cities.iloc[-1]} customers. Near-unique, unrealistic geographic distribution; "
            f"demonstrates the technique, not a real concentration signal.",
        )

    with col4:
        acc_ct = accounts.groupby("customer_id").size().rename("account_count")
        card_ct = (
            accounts[["account_id", "customer_id"]]
            .merge(cards[["account_id"]], on="account_id")
            .groupby("customer_id").size().rename("card_count")
        )
        loan_ct = loans.groupby("customer_id").size().rename("loan_count")
        prod = pd.concat([acc_ct, card_ct, loan_ct], axis=1)
        prod_full = customers[["customer_id"]].merge(prod, left_on="customer_id", right_index=True, how="left").fillna(0)

        dist = prod_full["account_count"].value_counts().sort_index()
        fig4 = px.bar(x=dist.index.astype(int), y=dist.values,
                      labels={"x": "Number of Accounts", "y": "Customers"},
                      title="Accounts per Customer", color_discrete_sequence=[BERRY])
        st.plotly_chart(fig4, use_container_width=True)
        zero_acc = (prod_full["account_count"] == 0).sum()
        callout(
            f"{zero_acc:,} customers ({100*zero_acc/len(prod_full):.1f}%) hold <b>zero</b> accounts "
            f"at all. Any KPI implicitly assuming 'every customer has an account' needs to account "
            f"for this — a naive per-customer average balance would silently exclude over a fifth "
            f"of the customer base.",
            kind="finding",
        )

# ============================================================ AREA 2
elif area.startswith("2"):
    st.subheader("Area 2 — Account Portfolio")

    bal_stats = accounts.groupby("account_type")["balance_usd"].agg(
        mean="mean", median="median",
        p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90), count="count",
    ).round(2)
    st.dataframe(bal_stats, use_container_width=True)
    callout(
        "Mean/median/P10/P90 are nearly identical across Checking, Savings, and Business "
        "(~$100K mean, ~$20K-$180K P10-P90 in all three) — account_type has essentially no "
        "relationship to balance size.",
        kind="finding",
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(accounts, x="account_type", y="balance_usd", color="account_type",
                     title="Balance Distribution by Account Type", points=False,
                     color_discrete_sequence=PLOTLY_SEQUENCE)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        monthly = accounts.groupby("open_month").agg(
            accounts_opened=("account_id", "count"), avg_balance=("balance_usd", "mean")
        ).reset_index()
        fig2 = px.line(monthly, x="open_month", y="accounts_opened",
                       title="Accounts Opened by Month (2019-2025)", color_discrete_sequence=[NAVY])
        st.plotly_chart(fig2, use_container_width=True)

    branch_aum = (
        accounts.merge(branches[["branch_id", "branch_name"]], on="branch_id")
        .groupby(["branch_id", "branch_name"])
        .agg(account_count=("account_id", "count"), total_aum=("balance_usd", "sum"))
        .sort_values("total_aum", ascending=False).reset_index()
    )
    branch_aum["cumulative_pct"] = 100 * branch_aum["total_aum"].cumsum() / branch_aum["total_aum"].sum()
    top30 = branch_aum.head(30)
    fig3 = px.bar(top30, x="branch_name", y="total_aum", title="Branch AUM Pareto — Top 30 of 500 Branches",
                  color_discrete_sequence=[TEAL])
    fig3.update_xaxes(tickangle=60, title="")
    fig3.update_yaxes(title="Total AUM (USD)")
    st.plotly_chart(fig3, use_container_width=True)

    n20 = int(len(branch_aum) * 0.2)
    pct20 = 100 * branch_aum["total_aum"].head(n20).sum() / branch_aum["total_aum"].sum()
    callout(
        f"Top 20% of branches ({n20} of {len(branch_aum)}) hold {pct20:.1f}% of total AUM — a "
        f"moderate, not extreme, concentration, consistent with the engineered lognormal-weighted "
        f"branch assignment (seed=42, sigma=0.6). Largest branch: <b>{branch_aum.iloc[0]['branch_name']}</b> "
        f"with {branch_aum.iloc[0]['account_count']:,} accounts, ${branch_aum.iloc[0]['total_aum']:,.0f} AUM.",
    )

# ============================================================ AREA 3
elif area.startswith("3"):
    st.subheader("Area 3 — Card Portfolio")

    col1, col2 = st.columns(2)
    with col1:
        split = cards["card_type"].value_counts()
        fig = px.pie(values=split.values, names=split.index, title="Overall Debit vs Credit Split",
                     color_discrete_sequence=[NAVY, TEAL])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ac = cards.merge(accounts[["account_id", "account_type"]], on="account_id")
        ct = pd.crosstab(ac["account_type"], ac["card_type"], normalize="index") * 100
        fig2 = px.bar(ct, barmode="stack", title="Card Type Split by Account Type (%)",
                      color_discrete_sequence=[TEAL, NAVY])
        st.plotly_chart(fig2, use_container_width=True)
        callout("Split is ~50/50 both overall and within every account_type — uniform random "
                "assignment, no meaningful relationship between account_type and card_type.")

    today = pd.Timestamp.now().normalize()
    days_to_exp = (cards["expiration_date"] - today).dt.days

    def exp_window(d):
        if d < 0: return "4. Already expired"
        elif d <= 30: return "1. Within 30 days"
        elif d <= 60: return "2. Within 60 days"
        elif d <= 90: return "3. Within 90 days"
        else: return "5. More than 90 days out"

    cards["exp_window"] = days_to_exp.apply(exp_window)
    exp_counts = cards["exp_window"].value_counts().sort_index()
    labels = [s.split(". ", 1)[1] for s in exp_counts.index.astype(str)]
    fig3 = px.bar(x=labels, y=exp_counts.values, title="Card Expiration Windows",
                  color_discrete_sequence=[PURPLE])
    st.plotly_chart(fig3, use_container_width=True)
    already_expired = int((days_to_exp < 0).sum())
    callout(
        f"~{already_expired:,} cards (~{100*already_expired/len(cards):.1f}%, roughly 1 in 5) are "
        f"already expired as of today — this count shifts daily and should never be quoted to the "
        f"exact digit. No card-status field exists to exclude them from portfolio KPIs.",
        kind="finding",
    )

# ============================================================ AREA 4
elif area.startswith("4"):
    st.subheader("Area 4 — Loan Book")

    pct = loans["loan_amount"].quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (p, label) in zip([c1, c2, c3, c4, c5], zip(pct.index, ["P25", "P50", "P75", "P90", "P99"])):
        col.metric(label, f"${pct[p]:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(loans, x="loan_amount", nbins=50, title="Loan Amount Distribution",
                           color_discrete_sequence=[NAVY])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        corr = loans["loan_amount"].corr(loans["interest_rate"])
        fig2 = px.scatter(loans.sample(min(20000, len(loans)), random_state=1),
                          x="loan_amount", y="interest_rate", opacity=0.25,
                          title=f"Loan Amount vs Interest Rate (r = {corr:.4f})",
                          color_discrete_sequence=[BERRY])
        st.plotly_chart(fig2, use_container_width=True)
        callout(
            f"Correlation is essentially zero (r = {corr:.4f}) — a formless cloud, exactly what "
            f"'no relationship' looks like. Directly foreshadows why the predictive model's target "
            f"has to be <b>constructed</b>, not learned naturally.",
            kind="finding",
        )

    lc = loans.merge(customers[["customer_id", "credit_band"]], on="customer_id")
    band_summary = lc.groupby("credit_band", observed=True).agg(
        loan_count=("loan_amount", "size"), avg_amount=("loan_amount", "mean"), avg_rate=("interest_rate", "mean")
    ).round(2)
    band_summary.index = [s.split(". ", 1)[1] for s in band_summary.index.astype(str)]
    st.dataframe(band_summary, use_container_width=True)
    callout(
        f"Avg loan amount (~$148.7K-$151.1K) and avg interest rate (8.39%-8.64%) are essentially "
        f"flat across every credit band — a customer with 'Poor' credit pays almost the same "
        f"average rate as one with 'Exceptional' credit in this dataset.",
        kind="finding",
    )

    orig_month = loans.groupby("origination_month").size().reset_index(name="loans_originated")
    fig3 = px.line(orig_month, x="origination_month", y="loans_originated",
                   title="Loan Originations by Month (2019-2025)", color_discrete_sequence=[TEAL])
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"Monthly range: {orig_month['loans_originated'].min()}-{orig_month['loans_originated'].max()} "
        f"loans/month — flat at every granularity checked, no growth trend."
    )

# ============================================================ AREA 5
else:
    st.subheader("Area 5 — Branch Network")

    branch_sizes = accounts.groupby("branch_id").size()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Smallest branch", f"{branch_sizes.min()} accounts")
    c2.metric("Largest branch", f"{branch_sizes.max():,} accounts")
    c3.metric("Mean", f"{branch_sizes.mean():.1f}")
    c4.metric("Median", f"{branch_sizes.median():.1f}")

    fig = px.histogram(branch_sizes, nbins=40, title="Branch Size Distribution (Accounts per Branch)",
                       color_discrete_sequence=[TEAL])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    callout(
        "Right-skewed as designed — the engineered lognormal-weighted branch assignment "
        "(seed=42, sigma=0.6) deliberately avoided a flat/uniform split to look more realistic.",
    )

    acc_primary = accounts.sort_values("balance_usd", ascending=False).drop_duplicates("customer_id")
    cust_branch = acc_primary[["customer_id", "branch_id"]]
    loans_branch = loans.merge(cust_branch, on="customer_id", how="left")
    branch_deposits = accounts.groupby("branch_id")["balance_usd"].sum().rename("total_deposits")
    branch_loan_amt = loans_branch.groupby("branch_id")["loan_amount"].sum().rename("total_loans_attributed")
    combo = pd.concat([branch_deposits, branch_loan_amt], axis=1).fillna(0).sort_values("total_deposits", ascending=False)
    top20 = combo.head(20).reset_index()
    fig2 = px.bar(top20, x="branch_id", y=["total_deposits", "total_loans_attributed"], barmode="group",
                 title="Deposits vs. Attributed Loans — Top 20 Branches by Deposits",
                 color_discrete_sequence=[NAVY, PURPLE])
    fig2.update_xaxes(tickangle=60, title="")
    st.plotly_chart(fig2, use_container_width=True)
    callout(
        "Loans have no direct branch_id — attributed here to each customer's PRIMARY account "
        "(highest balance). This is a disclosed analytical assumption, not a data field.",
    )
