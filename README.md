# 🏦 PRP Project 2 — Banking Analytics Platform

**An end-to-end, live-computed banking analytics web app** covering fraud risk, credit risk, and
branch performance for a synthetic 1.26M-row retail/commercial banking dataset — built as a
data-analyst internship capstone and deployed as an interactive Streamlit application.

🔗 **Live app:** _add your Streamlit Community Cloud URL here after deployment_

---

## Overview

This app is the final, interactive deployment stage of a full analytics pipeline: raw data →
SQL data-quality checks and cleaning → KPI definitions → descriptive, diagnostic, and predictive
analysis in Python → Power BI-style dashboards — all rebuilt here as one Streamlit application so
that every number on every page is **calculated live from the underlying data**, not pulled from a
static screenshot or a cached export.

**Business problem:** an anonymized retail/commercial bank has no systematic, data-driven view of
(1) transaction-level fraud risk, where no ground-truth fraud label exists in the data, and (2)
loan default risk, where no model currently flags at-risk loans before they default. This project
builds a transparent, rule-based fraud-detection system, a single flagship logistic-regression
default-risk model, and four stakeholder-specific dashboards to close that gap.

## Key features

- **9 fully interactive pages** — from raw-data exploration through to a live loan-risk scoring
  widget — all built on real pandas computation, not pre-rendered charts.
- **Transparent fraud detection** — a 4-rule, fully explainable scoring system (amount anomaly,
  transaction velocity, merchant diversity, geographic improbability), including the evidence for
  why two candidate rules were deliberately rejected.
- **A defensible predictive model** — a `statsmodels` logistic regression for loan default risk,
  built through a documented, reproducible workflow (Somers' D feature selection, VIF collinearity
  checks, 3 model iterations, cutoff optimization, decile validation).
- **Four stakeholder dashboards** — Branch Operations, Credit Risk, Fraud & Compliance, and
  Executive Overview, each scoped to what that audience actually needs to act on.
- **Radical honesty about data quality** — a dedicated page disclosing every synthetic, engineered,
  or constructed element in the dataset, so every number elsewhere is read in the right context.

## Tech stack

| Layer | Tool |
|---|---|
| Data cleaning & KPIs | SQL (Snowflake) |
| Analysis & modeling | Python (pandas, statsmodels, scikit-learn) — developed in Databricks |
| Predictive model | Logistic Regression (`statsmodels.formula.api.logit`) |
| Dashboards (prior deliverable) | Power BI |
| Web application | **Streamlit** |
| Version control / hosting | GitHub + Streamlit Community Cloud |

## Pages

| Page | What it shows |
|---|---|
| **Home** | Headline KPIs, the "Independence Story" finding, and recommendations for management |
| **Descriptive Analysis** | Customer base, account portfolio, card portfolio, loan book, branch network |
| **Diagnostic Analysis** | The 4-rule fraud-scoring system (with rejected-rule evidence) and root-cause diagnostics |
| **Predictive Analysis** | The full model-build workflow, validation metrics, a **live risk-scoring widget**, and the scored loan book |
| **Branch Operations** | Branch-manager view: AUM, deposit trend, account mix, branch drill-through |
| **Credit Risk** | Credit-risk-team view: model output by risk tier, score distribution, origination trend |
| **Fraud & Compliance** | Compliance view: flagged-transaction queue, time/rule heatmaps, account drill-through |
| **Executive Overview** | Leadership view: bank-wide health in one page |
| **Data Quality & Limitations** | Full, honest disclosure of every synthetic/engineered/constructed element in the dataset |

## Dataset

A synthetic banking dataset spanning 7 related tables and **1,260,500 total rows**:

| Table | Rows | Table | Rows |
|---|---|---|---|
| Customers | 50,000 | Merchants | 5,000 |
| Accounts | 75,000 | Branches | 500 |
| Cards | 100,000 | Transactions | 1,000,000 |
| Loans | 30,000 | | |

## Running locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens directly in your browser at `http://localhost:8501` — no login required.

> **Note:** the first visit to the Diagnostic, Predictive, Fraud & Compliance, or Executive
> Overview pages takes 30–90 seconds (scoring 1,000,000 transactions for fraud risk and training
> the logistic regression model). Both are cached with `st.cache_data` afterward for the rest of
> that session.

## Deployment

This app is deployed on **Streamlit Community Cloud**, pointed at `streamlit_app.py` as the main
file. No database or external service is required — all data ships inside `data/` as CSVs, and
`requirements.txt` is installed automatically by the platform.

## Project structure

```
.
├── streamlit_app.py              # Home page (entry point)
├── pages/                        # One file per page — Streamlit auto-builds the sidebar nav
│   ├── 1_Descriptive_Analysis.py
│   ├── 2_Diagnostic_Analysis.py
│   ├── 3_Predictive_Analysis.py
│   ├── 4_Branch_Operations.py
│   ├── 5_Credit_Risk.py
│   ├── 6_Fraud_Compliance.py
│   ├── 7_Executive_Overview.py
│   └── 8_Data_Quality_Limitations.py
├── utils/
│   ├── style.py                  # Shared color palette and CSS
│   ├── data_loader.py            # Loads, cleans, and flags the 7 raw tables
│   ├── fraud.py                  # The 4-rule fraud-scoring engine
│   └── model.py                  # The flagship logistic regression (build + live scoring)
├── data/                         # The 7 raw CSVs (transactions split into 4 parts)
├── .streamlit/
│   └── config.toml               # App theme
└── requirements.txt
```

## Known limitations

See the in-app **Data Quality & Limitations** page for the full disclosure. In short:

- `derived_fraud_score` and `derived_default_risk_flag` are both constructed, rule-based labels —
  no ground-truth outcome exists in this dataset for either fraud or loan default.
- `accounts.branch_id` is an engineered, seeded-random assignment; `branches.city`/`country` are
  synthetic and have no real correspondence to an actual branch location.
- Roughly half of accounts, loans, and transactions carry a timeline-logic inconsistency (the
  activity date precedes the parent record's creation date) — flagged throughout via
  `date_logic_flag`, never silently dropped.

## License

Released under the [MIT License](LICENSE) — this is a portfolio/academic project built on a
synthetic dataset; no real customer data is included or represented.

## Author

Built as a data analyst internship capstone project.
