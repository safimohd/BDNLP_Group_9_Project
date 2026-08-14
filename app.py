import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SteelFlow Logistics | Multi-Tier Forecasting Ecosystem",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ------------------------------------------------------------------------------
# DATA LOADERS WITH STREAMLIT CACHING
# ------------------------------------------------------------------------------
@st.cache_data
def load_core_data():
    norm_df = pd.read_csv(os.path.join(DATA_DIR, "weekly_demand_normalized.csv")).set_index("relative_week")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "weekly_demand_train.csv")).set_index("relative_week")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "weekly_demand_test.csv")).set_index("relative_week")
    return norm_df, train_df, test_df

@st.cache_data
def load_benchmarks():
    three_way = pd.read_csv(os.path.join(DATA_DIR, "three_way_model_benchmark.csv"))
    scorecard = pd.read_csv(os.path.join(DATA_DIR, "vertical_prioritisation_scorecard.csv"))
    granger = pd.read_csv(os.path.join(DATA_DIR, "granger_centrality.csv"))
    
    with open(os.path.join(DATA_DIR, "lstm_results.json"), "r") as f:
        lstm = json.load(f)
    with open(os.path.join(DATA_DIR, "prophet_results.json"), "r") as f:
        prophet = json.load(f)
    with open(os.path.join(DATA_DIR, "var_results.json"), "r") as f:
        var = json.load(f)
        
    cold_start = pd.read_csv(os.path.join(DATA_DIR, "cold_start_nlp_benchmark_full_3.4M.csv"))
    return three_way, scorecard, granger, lstm, prophet, var, cold_start

norm_df, train_df, test_df = load_core_data()
three_way, scorecard, granger, lstm_results, prophet_results, var_results, cold_start_summary = load_benchmarks()
depts = sorted(norm_df.columns.tolist())

# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/delivery-conveyer-belt.png", width=70)
st.sidebar.title("SteelFlow Intelligence")
st.sidebar.caption("Big Data Demand Forecasting Platform (3.42M Transactions)")

view = st.sidebar.radio(
    "Select Decision View:",
    [
        "1. Executive Overview & KPIs",
        "2. Timeline & Demand Normalization",
        "3. Three-Way Model Leaderboard",
        "4. LSTM Neural Network Diagnostics",
        "5. Granger Causality Demand Network",
        "6. Multivariate Cluster VAR System",
        "7. Vertical Prioritization Scorecard",
        "8. Cold-Start SKU NLP Engine",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Dataset:** Full Population (3.42M Orders, Weeks 0–51, 19 Verticals)")

# ------------------------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------------------------
if view == "1. Executive Overview & KPIs":
    st.title("📦 SteelFlow Logistics: Executive Demand Ecosystem")
    st.markdown("### Production-Grade Forecasting & Network Intelligence Architecture")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transaction Volume", "3.42 Million Orders", "100% Population")
    col2.metric("Mean LSTM Forecast MAPE", "5.43%", "-0.89% vs ARIMA")
    col3.metric("Granger Demand Links", "117 Significant Pairs", "p < 0.05")
    col4.metric("Catalogue Scope", "19 Departments", "Weeks 0–51")
    
    st.markdown("---")
    c_left, c_right = st.columns([3, 2])
    
    with c_left:
        st.subheader("Top Priority Supply-Chain Verticals")
        top_sc = scorecard.head(6)[["priority_rank", "department", "final_priority_score", "forecast_reliability"]]
        st.dataframe(top_sc.style.format({"final_priority_score": "{:.4f}"}), use_container_width=True)
    
    with c_right:
        st.subheader("Model Championship Distribution")
        fig = px.pie(
            three_way, names="best_model", 
            title="Winning Model Breakdown (19 Verticals)",
            color="best_model",
            color_discrete_map={"LSTM": "#2563EB", "ARIMA": "#10B981", "PROPHET": "#F59E0B"}
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 2: TIMELINE & NORMALIZATION
# ------------------------------------------------------------------------------
elif view == "2. Timeline & Demand Normalization":
    st.title("📈 Timeline Reconstruction & Macro Exposure Normalization")
    st.markdown("Reconstructing the user-relative lifecycle timeline to eliminate synthetic cohort onboarding decay.")
    
    selected_dept = st.selectbox("Select Department to Inspect Normalized Series:", depts, index=depts.index("produce"))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df[selected_dept], mode="lines+markers", name="Training Split (Wks 0–42)", line=dict(color="#3B82F6")))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df[selected_dept], mode="lines+markers", name="Evaluation Horizon (Wks 43–51)", line=dict(color="#EF4444")))
    fig.add_vline(x=42.5, line_dash="dash", line_color="orange", annotation_text="Train / Test Split Boundary")
    fig.update_layout(title=f"Demand per Active User: {selected_dept.upper()}", xaxis_title="Relative Week (0–51)", yaxis_title="Normalized Weekly Demand")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 3: THREE-WAY MODEL LEADERBOARD
# ------------------------------------------------------------------------------
elif view == "3. Three-Way Model Leaderboard":
    st.title("🏆 Three-Way Forecasting Benchmark (Weeks 43–51 Test Split)")
    st.markdown("Head-to-head performance audit: **Auto-ARIMA vs. Facebook Prophet vs. Walk-Forward LSTM**.")
    
    st.dataframe(three_way.style.highlight_min(subset=["arima_mape", "prophet_mape", "lstm_mape"], axis=1, color="#1E3A8A"), use_container_width=True)
    
    fig = px.bar(
        three_way, x="department", y=["arima_mape", "prophet_mape", "lstm_mape"],
        barmode="group", title="MAPE (%) Comparison across 19 Departments",
        labels={"value": "MAPE (%)", "variable": "Model Architecture"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 4: LSTM DIAGNOSTICS
# ------------------------------------------------------------------------------
elif view == "4. LSTM Diagnostics":
    st.title("🧠 Walk-Forward Keras LSTM Forecaster")
    dept_choice = st.selectbox("Select Department:", depts, index=depts.index("frozen"))
    
    d_res = lstm_results[dept_choice]
    eval_weeks = list(range(43, 52))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eval_weeks, y=d_res["actual"], mode="lines+markers", name="Actual Ground Truth", line=dict(color="#10B981", width=3)))
    fig.add_trace(go.Scatter(x=eval_weeks, y=d_res["preds"], mode="lines+markers", name="LSTM Walk-Forward Prediction", line=dict(color="#6366F1", width=3, dash="dash")))
    fig.update_layout(title=f"{dept_choice.upper()} — Test Forecast (MAPE: {d_res['mape']}%, RMSE: {d_res['rmse']})", xaxis_title="Evaluation Week", yaxis_title="Demand")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 5: GRANGER CAUSALITY NETWORK
# ------------------------------------------------------------------------------
elif view == "5. Granger Causality Network":
    st.title("🌐 Pairwise Granger Causality & Lead-Lag Network")
    st.markdown("Evaluated on differenced series ($p < 0.05$) across full 3.42M transaction history.")
    
    fig = px.bar(
        granger.sort_values("net_leadership", ascending=False),
        x="department", y="net_leadership",
        color="net_leadership", color_continuous_scale="Blues",
        title="Net Leadership Metric (Out-Degree minus In-Degree)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(granger, use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 6: MULTIVARIATE CLUSTER VAR
# ------------------------------------------------------------------------------
elif view == "6. Multivariate Cluster VAR System":
    st.title("📈 Cluster Vector Autoregression System: `Alcohol-Beverages-Snacks-Breakfast`")
    st.markdown("Capturing inter-category cross-elasticity and co-movement feedback loops.")
    
    var_df = pd.DataFrame(var_results).T
    st.dataframe(var_df.style.format({"rmse": "{:.5f}", "mape": "{:.2f}%"}), use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 7: DECISION SCORECARD
# ------------------------------------------------------------------------------
elif view == "7. Vertical Prioritization Scorecard":
    st.title("🎖️ Multi-Factor Vertical Prioritization Decision Scorecard")
    st.markdown("$$\\text{Score} = (0.35\\times\\text{Vol} + 0.25\\times\\text{Acc} + 0.20\\times\\text{Stab} + 0.20\\times\\text{Cent}) \\times \\text{Reliability Multiplier}$$")
    
    st.dataframe(scorecard, use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 8: COLD-START NLP ENGINE
# ------------------------------------------------------------------------------
elif view == "8. Cold-Start SKU NLP Engine":
    st.title("🔍 TF-IDF Semantic Cold-Start SKU Forecaster")
    st.markdown("Benchmarking 4 Estimators for New Item Introductions across 3.42M orders.")
    
    st.dataframe(cold_start_summary.style.highlight_min(subset=["MAE", "RMSE", "WAPE (%)"], color="#1E3A8A"), use_container_width=True)