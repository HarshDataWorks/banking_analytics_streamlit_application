"""
Diagnostic-phase fraud scoring — a direct port of 07_diagnostic_analysis.py's
Section 3 (the exact same four rules, in the exact same order, with the exact
same thresholds). Nothing here is new logic; this module only swaps the
Databricks/Snowflake load for a CSV load, the same substitution pattern used
to verify that notebook against the raw CSVs in earlier project phases.

Standing constraint, carried over from the diagnostic phase and never
relaxed: percentiles, ratios, group averages, and plain counts only. No
z-scores, no p-values, no ANOVA, no regression here — a transaction's
`derived_fraud_score` is just how many of the 4 rules it breaks (0-4),
counted, not weighted or modeled.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_transactions, load_merchants


def _count_nearby_transactions(times: np.ndarray, window_minutes: int = 10) -> np.ndarray:
    """For every transaction in a single account's sorted history, count how
    many of that SAME account's other transactions happened within
    `window_minutes` before or after it (inclusive of itself)."""
    n = len(times)
    window = np.timedelta64(window_minutes, "m")
    lo = np.searchsorted(times, times - window, side="left")
    hi = np.searchsorted(times, times + window, side="right")
    return hi - lo


def _count_recent_distinct_merchants(times: np.ndarray, merch: np.ndarray, window_days: int = 30) -> np.ndarray:
    """For every transaction, count how many DISTINCT merchants that same
    account used in the trailing `window_days` days (including itself)."""
    n = len(times)
    window = np.timedelta64(window_days, "D")
    lo = np.searchsorted(times, times - window, side="left")
    out = np.empty(n, dtype=int)
    for i in range(n):
        out[i] = len(set(merch[lo[i] : i + 1]))
    return out


@st.cache_data(show_spinner="Scoring 1,000,000 transactions for fraud risk (one-time, ~30-60s)...")
def compute_fraud_scores() -> pd.DataFrame:
    transactions = load_transactions().copy()
    merchants = load_merchants()[["merchant_id", "city"]].rename(columns={"city": "merchant_city"})

    # --- Rule 1: Amount anomaly — more than double this account's own average spend ---
    acc_avg_amt = transactions.groupby("account_id")["amount_usd"].mean().rename("acc_avg_amt")
    transactions = transactions.merge(acc_avg_amt, on="account_id", how="left")
    transactions["ratio_to_own_avg"] = transactions["amount_usd"] / transactions["acc_avg_amt"]
    transactions["flag_amount_anomaly"] = (transactions["ratio_to_own_avg"] > 2.0).astype(int)

    # --- Sort once, by account then time, for the two time-window rules below ---
    transactions = transactions.sort_values(["account_id", "transaction_date"]).reset_index(drop=True)

    # --- Rule 2: Velocity anomaly — 3+ of this account's own transactions within 10 minutes ---
    nearby_counts = []
    for _, g in transactions.groupby("account_id", sort=False):
        times = g["transaction_date"].values.astype("datetime64[ns]")
        nearby_counts.append(pd.Series(_count_nearby_transactions(times), index=g.index))
    transactions["nearby_txn_count_10min"] = pd.concat(nearby_counts).sort_index()
    transactions["flag_velocity_anomaly"] = (transactions["nearby_txn_count_10min"] >= 3).astype(int)

    # --- Rule 3: Merchant-diversity anomaly — 3+ distinct merchants in the trailing 30 days ---
    diversity_counts = []
    for _, g in transactions.groupby("account_id", sort=False):
        times = g["transaction_date"].values.astype("datetime64[ns]")
        merch = g["merchant_id"].values
        diversity_counts.append(pd.Series(_count_recent_distinct_merchants(times, merch), index=g.index))
    transactions["distinct_merchants_last_30days"] = pd.concat(diversity_counts).sort_index()
    transactions["flag_merchant_diversity_anomaly"] = (
        transactions["distinct_merchants_last_30days"] >= 3
    ).astype(int)

    # --- Rule 4: Geo improbability — different city than the previous transaction, within 60 min ---
    transactions = transactions.merge(merchants, on="merchant_id", how="left")
    transactions = transactions.sort_values(["account_id", "transaction_date"]).reset_index(drop=True)
    prev_city = transactions.groupby("account_id")["merchant_city"].shift(1)
    prev_time = transactions.groupby("account_id")["transaction_date"].shift(1)
    gap_minutes = (transactions["transaction_date"] - prev_time).dt.total_seconds() / 60
    transactions["flag_geo_improbability"] = (
        (gap_minutes <= 60) & (transactions["merchant_city"] != prev_city) & prev_city.notna()
    ).astype(int)

    # --- Composite score: a plain count of the 4 rules, 0-4. No weights. ---
    flag_columns = [
        "flag_amount_anomaly",
        "flag_velocity_anomaly",
        "flag_merchant_diversity_anomaly",
        "flag_geo_improbability",
    ]
    transactions["derived_fraud_score"] = transactions[flag_columns].sum(axis=1)

    def tier(n):
        if n == 0:
            return "Low"
        elif n == 1:
            return "Medium"
        else:
            return "High"

    transactions["fraud_tier"] = transactions["derived_fraud_score"].apply(tier)

    keep_cols = [
        "transaction_id", "account_id", "merchant_id", "amount_usd", "transaction_date",
        "hour", "day_of_week", "transaction_month", "date_logic_flag",
        "flag_amount_anomaly", "flag_velocity_anomaly",
        "flag_merchant_diversity_anomaly", "flag_geo_improbability",
        "derived_fraud_score", "fraud_tier",
    ]
    return transactions[keep_cols]


@st.cache_data(show_spinner=False)
def fraud_rejected_rules_evidence() -> dict:
    """Re-derives the two negative findings from 07_diagnostic_analysis.py
    Section 3.2 and 3.3 — the time-of-day rule that was tried and rejected,
    and the velocity rule's near-zero hit rate explained by typical gap size.
    Kept separate from the main scoring function since these never feed
    `derived_fraud_score` — they're evidence for *why* two rules aren't used."""
    transactions = load_transactions()
    hour_counts = transactions["hour"].value_counts().sort_index()
    would_flag_oddhour = transactions["hour"].between(0, 4).sum()

    gap_minutes = (
        transactions.sort_values(["account_id", "transaction_date"])
        .groupby("account_id")["transaction_date"]
        .diff()
        .dt.total_seconds()
        / 60
    )
    return {
        "hour_counts": hour_counts,
        "oddhour_flagged": int(would_flag_oddhour),
        "oddhour_pct": 100 * would_flag_oddhour / len(transactions),
        "median_gap_minutes": float(gap_minutes.median()),
        "median_gap_days": float(gap_minutes.median() / (60 * 24)),
    }
