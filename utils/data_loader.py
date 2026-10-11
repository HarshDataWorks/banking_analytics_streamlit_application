"""
Data loading, cleaning, and flagging layer for the PRP Project 2 Streamlit app.

This module is the Python/pandas equivalent of 03_data_quality_checks.sql +
04_cleaning_transformations.sql — it loads the 7 raw tables and adds the same
two disclosure flags that already exist on the live Snowflake database:

  - date_logic_flag            (on accounts, loans, transactions)
  - city_country_synthetic_flag (on branches, constant TRUE)

Cleaning philosophy, carried over unchanged from the SQL phase: FLAG, never
DELETE. Nothing in this module drops a single row. Every KPI/chart in the
app computes on the FULL table by default, exactly like 05_kpi_definitions.sql
and 06_descriptive_eda.py did.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CREDIT_BAND_BINS = [300, 580, 670, 740, 800, 851]
CREDIT_BAND_LABELS = [
    "1. Poor (300-579)",
    "2. Fair (580-669)",
    "3. Good (670-739)",
    "4. Very Good (740-799)",
    "5. Exceptional (800-850)",
]
CREDIT_BAND_LABELS_SHORT = ["Poor", "Fair", "Good", "Very Good", "Exceptional"]


def _credit_band(credit_score_series: pd.Series) -> pd.Series:
    return pd.cut(
        credit_score_series,
        bins=CREDIT_BAND_BINS,
        labels=CREDIT_BAND_LABELS,
        right=False,
    )


@st.cache_data(show_spinner="Loading customers...")
def load_customers() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "customers.csv", encoding="utf-8-sig")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["credit_band"] = _credit_band(df["credit_score"])
    today = pd.Timestamp.now().normalize()
    tenure_years = (today - df["created_at"]).dt.days / 365.25
    df["tenure_years"] = tenure_years

    def cohort(y):
        if y < 1:
            return "1. 0-1 yr"
        elif y < 3:
            return "2. 1-3 yr"
        elif y < 5:
            return "3. 3-5 yr"
        else:
            return "4. 5yr+"

    df["tenure_cohort"] = tenure_years.apply(cohort)
    return df


@st.cache_data(show_spinner="Loading accounts...")
def load_accounts() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "accounts.csv", encoding="utf-8-sig")
    df["open_date"] = pd.to_datetime(df["open_date"])

    customers = load_customers()[["customer_id", "created_at"]]
    merged = df.merge(customers, on="customer_id", how="left")
    df["date_logic_flag"] = (merged["open_date"] < merged["created_at"]).values

    df["open_month"] = df["open_date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    return df


@st.cache_data(show_spinner="Loading cards...")
def load_cards() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cards.csv", encoding="utf-8-sig")
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])
    return df


@st.cache_data(show_spinner="Loading loans...")
def load_loans() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "loans.csv", encoding="utf-8-sig")
    df["start_date"] = pd.to_datetime(df["start_date"])

    customers = load_customers()[["customer_id", "created_at"]]
    merged = df.merge(customers, on="customer_id", how="left")
    df["date_logic_flag"] = (merged["start_date"] < merged["created_at"]).values

    df["origination_month"] = df["start_date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    return df


@st.cache_data(show_spinner="Loading merchants...")
def load_merchants() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "merchants.csv", encoding="utf-8-sig")


@st.cache_data(show_spinner="Loading branches...")
def load_branches() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "branches.csv", encoding="utf-8-sig")
    df["city_country_synthetic_flag"] = True
    return df


@st.cache_data(show_spinner="Loading transactions (1,000,000 rows, 4 parts)...")
def load_transactions() -> pd.DataFrame:
    parts = []
    for i in range(1, 5):
        p = pd.read_csv(
            DATA_DIR / f"transactions_part{i}.csv",
            encoding="utf-8-sig",
        )
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    # Raw format is DD-MM-YYYY HH:MM (confirmed in the SQL-phase date-parsing checks)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True)

    accounts = load_accounts()[["account_id", "open_date"]]
    merged = df.merge(accounts, on="account_id", how="left")
    df["date_logic_flag"] = (merged["transaction_date"] < merged["open_date"]).values

    df["hour"] = df["transaction_date"].dt.hour
    df["day_of_week"] = df["transaction_date"].dt.day_name()
    df["transaction_month"] = df["transaction_date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    return df


def totals() -> dict:
    """Quick, cheap headline counts used on the Home page."""
    return {
        "customers": 50_000,
        "accounts": 75_000,
        "cards": 100_000,
        "loans": 30_000,
        "merchants": 5_000,
        "branches": 500,
        "transactions": 1_000_000,
        "total_rows": 1_260_500,
    }
