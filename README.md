# 📦 SteelFlow Logistics: Multi-Tier Demand Forecasting & Network Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, end-to-end demand forecasting, econometric network modeling, and cold-start SKU intelligence ecosystem built for **SteelFlow Logistics**. Evaluated across **3,417,064 transactions (33.8M line items)** and **47,879 catalogue SKUs** spanning 52 relative weeks and 19 commercial retail verticals.

---

## 🚀 Key System Capabilities

1. **Macro Exposure Normalization & Cohort Alignment**
   * Reconstructed user-relative purchase timelines across Weeks 0–51 to eliminate onboarding exposure decay.
   * Filtered uninformative placeholder departments (`missing`, `other`) and edge artefacts (Week 52).

2. **Three-Way Univariate Forecasting Benchmark**
   * Head-to-head out-of-sample evaluation on Weeks 43–51 (9-week test horizon).
   * **Models Benchmarked:** Auto-ARIMA vs. Facebook Prophet vs. Walk-Forward Deep Learning LSTM (tanh/relu).
   * **Results:** Walk-Forward LSTM captures complex non-linear co-movements to lead 10/19 verticals (Mean MAPE: **5.43%**), while Auto-ARIMA leads 9/19 verticals (Mean MAPE: **6.32%**).

3. **Econometric Network & Multi-Vertical Feedback Loops**
   * **Granger Causality:** 117 statistically significant lead-lag relationships ($p < 0.05$) across 342 pairwise vertical permutations.
   * **Multivariate Vector Autoregression (VAR):** Cluster-level modeling across the high-interaction `Alcohol-Beverages-Snacks-Breakfast` ecosystem.

4. **Strategic Vertical Prioritization Decision Scorecard**
   * Multi-criteria decision engine synthesizing Volume ($35\%$), Forecast Accuracy ($25\%$), Stability ($20\%$), and Network Centrality ($20\%$) scaled by a penalty-adjusted Forecast Reliability Multiplier.

5. **TF-IDF Semantic Cold-Start SKU Forecaster**
   * Solves zero-history inventory planning by extracting catalogue n-gram lexical similarities and matching new introductions to active analogue clusters.
   * Leverages robust Median Analogue estimation to protect against staple velocity inflation.

---

## 📊 Performance Summary

| Architecture / Model | Evaluation Scope | Mean MAPE (%) | Leading Verticals | Primary Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Walk-Forward LSTM** | 19 Verticals (Wks 43–51) | **5.43%** | **10 / 19** | High-velocity, non-linear demand verticals |
| **Auto-ARIMA** | 19 Verticals (Wks 43–51) | **6.32%** | **9 / 19** | Linear, autoregressive staple verticals |
| **Facebook Prophet** | 19 Verticals (Wks 43–51) | **9.97%** | **0 / 19** | Baseline additive trend decomposition |
| **TF-IDF Analogue Median** | 47,879 Catalogue SKUs | **2,262% WAPE** | Best Cold-Start | New item introductions (0 purchase history) |

---

## 🗂️ Repository Architecture

```text
steelflow_app/
│
├── app.py                             # Main 8-View Interactive Streamlit Dashboard
├── requirements.txt                   # Production dependencies (pinned for stability)
├── .gitignore                         # Git exclusion rules (venv, cache, temporary files)
├── README.md                          # Project documentation & architecture overview
│
├── .streamlit/
│   └── config.toml                    # UI Theme, dark-mode palette, and layout config
│
└── data/                              # Verified 100% Full-Dataset Artifacts (Output_New)
    ├── weekly_demand_normalized.csv   # Master 52-week normalized demand matrix (19 depts)
    ├── weekly_demand_train.csv        # Weeks 0–42 Training Split (43 time points)
    ├── weekly_demand_test.csv         # Weeks 43–51 Evaluation Horizon (9 time points)
    ├── adf_test_results.csv           # Augmented Dickey-Fuller stationarity tests
    ├── arima_results.csv              # Auto-ARIMA parameters, MAPE, and RMSE metrics
    ├── prophet_results.json           # Facebook Prophet 9-week horizon forecasts
    ├── lstm_results.json              # Walk-Forward LSTM predictions & test error metrics
    ├── three_way_model_benchmark.csv  # 3-Way Comparative Leaderboard & Champion assignments
    ├── granger_centrality.csv         # In-degree, out-degree, and net leadership network stats
    ├── var_results.json               # Cluster VAR(1) multivariate forecast system
    ├── vertical_prioritisation_scorecard.csv # Composite priority rankings (Ranks 1–19)
    └── cold_start_nlp_benchmark_full_3.4M.csv # 4-estimator Cold-Start SKU benchmark