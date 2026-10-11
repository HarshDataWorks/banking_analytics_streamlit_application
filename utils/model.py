"""
The flagship predictive model — Logistic Regression, Loan Default Risk.

This is a direct port of 08_predictive_analysis.py, using the exact same
toolkit the user's own class case study used: `smf.logit().fit()` (never
sklearn's LogisticRegression), Somers' D for feature importance, VIF for
multicollinearity, and a sensitivity+specificity scan for the cutoff. Only
the data source changed (CSV instead of Snowflake) and the final model is
re-fit here rather than re-loaded, since this app has no Snowflake
connection of its own.

Target: `derived_default_risk_flag` — a CONSTRUCTED label, never a real
observed default outcome (no such field exists in this dataset). Built from
Poor credit_score (<580) AND top-25% loan_amount OR interest_rate.
"""

import numpy as np
import pandas as pd
import streamlit as st

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from utils.data_loader import load_loans, load_customers


@st.cache_data(show_spinner="Training the flagship logistic regression model (one-time)...")
def build_model() -> dict:
    loans = load_loans()
    customers = load_customers()

    df = loans.merge(customers[["customer_id", "credit_score", "created_at"]], on="customer_id", how="left")
    df["loan_to_credit_ratio"] = df["loan_amount"] / df["credit_score"]
    df["tenure_at_origination_years"] = (df["start_date"] - df["created_at"]).dt.days / 365.25

    # --- Construct the target ---
    p75_amount = df["loan_amount"].quantile(0.75)
    p75_rate = df["interest_rate"].quantile(0.75)
    is_poor_credit = df["credit_score"] < 580
    is_top_amount = df["loan_amount"] > p75_amount
    is_top_rate = df["interest_rate"] > p75_rate
    df["derived_default_risk_flag"] = (is_poor_credit & (is_top_amount | is_top_rate)).astype(int)

    target_rate = df["derived_default_risk_flag"].mean()

    model_cols = [
        "credit_score", "loan_amount", "interest_rate",
        "loan_to_credit_ratio", "tenure_at_origination_years",
        "derived_default_risk_flag",
    ]
    model_df = df[model_cols].copy()

    # --- Outlier treatment: clip every numeric column at its 1st/99th percentile ---
    model_df = model_df.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))

    corr_matrix = model_df.corr()

    # --- Train/test split, 70:30, same random_state as the class case study ---
    train, test = train_test_split(model_df, test_size=0.3, random_state=10)

    # --- Model 1: all 5 candidate features ---
    features_m1 = train.columns.difference(["derived_default_risk_flag"])
    formula_m1 = "derived_default_risk_flag ~ " + " + ".join(features_m1)
    m1 = smf.logit(formula_m1, train).fit(disp=0)

    # --- Somers' D (bivariate AUC-based feature importance, cutoff 0.2) ---
    somers_rows = []
    for var_name in model_df.columns.difference(["derived_default_risk_flag"]):
        log_mod = smf.logit(formula=f"derived_default_risk_flag ~ {var_name}", data=model_df).fit(disp=0)
        y_pred = log_mod.predict(model_df)
        concordance = roc_auc_score(model_df.derived_default_risk_flag, y_pred)
        somers_rows.append({"variable": var_name, "roc_auc": concordance, "somers_d": 2 * concordance - 1})
    somers_df = pd.DataFrame(somers_rows).sort_values("somers_d", ascending=False)
    kept_features = somers_df.loc[somers_df.somers_d >= 0.2, "variable"].values

    # --- VIF check on the Somers'-D survivors ---
    eq = "derived_default_risk_flag ~ " + " + ".join(kept_features)
    _, b = dmatrices(eq, data=model_df, return_type="dataframe")
    vif_df = pd.DataFrame({
        "feature": b.columns,
        "VIF": [variance_inflation_factor(b.values, i) for i in range(b.shape[1])],
    })

    # --- Model 2: after the Somers' D cut ---
    m2 = smf.logit(eq, data=train).fit(disp=0)

    # --- Model 3: final model, after resolving the VIF issue (drop loan_to_credit_ratio) ---
    final_formula = "derived_default_risk_flag ~ credit_score + loan_amount + interest_rate"
    m3 = smf.logit(final_formula, data=train).fit(disp=0)

    _, b2 = dmatrices(final_formula, data=model_df, return_type="dataframe")
    vif_final = pd.DataFrame({
        "feature": b2.columns,
        "VIF": [variance_inflation_factor(b2.values, i) for i in range(b2.shape[1])],
    })

    coef = m3.params

    # --- Predict & evaluate ---
    train_predict = m3.predict(train)
    test_predict = m3.predict(test)
    train_auc = roc_auc_score(train.derived_default_risk_flag, train_predict)
    test_auc = roc_auc_score(test.derived_default_risk_flag, test_predict)

    train_pf = pd.DataFrame({"actual": train.derived_default_risk_flag.values, "prob": train_predict.values})
    test_pf = pd.DataFrame({"actual": test.derived_default_risk_flag.values, "prob": test_predict.values})

    # --- Best cutoff: maximize sensitivity + specificity over a 0..1 scan ---
    scan_rows = []
    for iproba in np.arange(0, 1.01, 0.01):
        y_pred = (train_pf.prob >= iproba).astype(int)
        cm = confusion_matrix(train_pf.actual, y_pred)[::-1, ::-1]
        tp, tn, fp, fn = cm[0, 0], cm[1, 1], cm[1, 0], cm[0, 1]
        row_tots = cm.sum(axis=1)
        tpr = tp / row_tots[0] if row_tots[0] else 0.0
        fpr = fp / row_tots[1] if row_tots[1] else 0.0
        specificity = 1 - fpr
        scan_rows.append({"proba": round(iproba, 2), "tpr": tpr, "specificity": specificity,
                           "sen_spec": tpr + specificity})
    scan_df = pd.DataFrame(scan_rows)
    best_cutoff = float(scan_df.loc[scan_df.sen_spec.idxmax(), "proba"])

    train_pf["predicted"] = (train_pf.prob >= best_cutoff).astype(int)
    test_pf["predicted"] = (test_pf.prob >= best_cutoff).astype(int)
    train_accuracy = accuracy_score(train_pf.actual, train_pf.predicted)
    test_accuracy = accuracy_score(test_pf.actual, test_pf.predicted)

    # --- Decile analysis (validation) ---
    train_pf["decile"] = pd.qcut(train_pf.prob, 10, labels=False, duplicates="drop")
    decile_summary = (
        train_pf.groupby("decile")
        .agg(min_prob=("prob", "min"), max_prob=("prob", "max"),
             actual_positive=("actual", "sum"), count=("actual", "count"))
        .reset_index()
        .sort_values("decile", ascending=False)
    )
    decile_summary["pct_positive"] = 100 * decile_summary["actual_positive"] / decile_summary["count"]

    # --- Odds ratios ---
    odds_credit_50 = float(np.exp(-coef["credit_score"] * 50))
    odds_loan_10k = float(np.exp(coef["loan_amount"] * 10000))
    odds_rate_1pt = float(np.exp(coef["interest_rate"] * 1))

    # --- Score the full loan book (un-clipped values, matching the notebook) ---
    df["prob"] = m3.predict(df)
    df["predicted_high_risk"] = (df["prob"] > best_cutoff).astype(int)

    def risk_tier(p):
        if p >= 0.5:
            return "3. High"
        elif p >= best_cutoff:
            return "2. Medium"
        else:
            return "1. Low"

    df["risk_tier"] = df["prob"].apply(risk_tier)

    scored_book = df[[
        "loan_id", "customer_id", "credit_score", "loan_amount", "interest_rate",
        "start_date", "origination_month", "derived_default_risk_flag",
        "prob", "predicted_high_risk", "risk_tier",
    ]].copy()

    return {
        "target_rate": float(target_rate),
        "p75_amount": float(p75_amount),
        "p75_rate": float(p75_rate),
        "corr_matrix": corr_matrix,
        "m1_summary": m1.summary2().as_text(),
        "somers_df": somers_df.reset_index(drop=True),
        "kept_features": list(kept_features),
        "vif_df": vif_df,
        "m2_summary": m2.summary2().as_text(),
        "m3_summary": m3.summary2().as_text(),
        "vif_final": vif_final,
        "coef": {"intercept": float(coef.get("Intercept", 0.0)),
                 "credit_score": float(coef["credit_score"]),
                 "loan_amount": float(coef["loan_amount"]),
                 "interest_rate": float(coef["interest_rate"])},
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "best_cutoff": best_cutoff,
        "scan_df": scan_df,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "decile_summary": decile_summary,
        "odds_credit_50": odds_credit_50,
        "odds_loan_10k": odds_loan_10k,
        "odds_rate_1pt": odds_rate_1pt,
        "scored_book": scored_book,
        "train_pf": train_pf,
        "test_pf": test_pf,
    }


def score_one_loan(coef: dict, credit_score: float, loan_amount: float, interest_rate: float) -> float:
    """Plain sigmoid scoring using the fitted Model 3 coefficients — used by
    the live scoring widget so it never needs to refit anything."""
    z = (
        coef["intercept"]
        + coef["credit_score"] * credit_score
        + coef["loan_amount"] * loan_amount
        + coef["interest_rate"] * interest_rate
    )
    return float(1 / (1 + np.exp(-z)))
