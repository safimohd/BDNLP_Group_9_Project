import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Set global plotly template
pio.templates.default = "plotly_white"

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SteelFlow Logistics | Demand Forecasting Ecosystem",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished Light Theme and KPI cards
st.markdown("""
<style>
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 4px;
        white-space: nowrap;
    }
    .kpi-delta-green {
        font-size: 0.8rem;
        font-weight: 600;
        color: #059669;
        background-color: #ECFDF5;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .kpi-delta-blue {
        font-size: 0.8rem;
        font-weight: 600;
        color: #2563EB;
        background-color: #EFF6FF;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ------------------------------------------------------------------------------
# DATA LOADERS WITH CACHING
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_time_series_data():
    norm_df = pd.read_csv(os.path.join(DATA_DIR, "weekly_demand_normalized.csv"))
    if "relative_week" in norm_df.columns:
        norm_df = norm_df.set_index("relative_week")
        
    train_df = pd.read_csv(os.path.join(DATA_DIR, "weekly_demand_train.csv"))
    if "relative_week" in train_df.columns:
        train_df = train_df.set_index("relative_week")
        
    test_df = pd.read_csv(os.path.join(DATA_DIR, "weekly_demand_test.csv"))
    if "relative_week" in test_df.columns:
        test_df = test_df.set_index("relative_week")
    return norm_df, train_df, test_df

@st.cache_data(show_spinner=False)
def get_tables_data():
    three_way = pd.read_csv(os.path.join(DATA_DIR, "three_way_model_benchmark.csv"))
    scorecard = pd.read_csv(os.path.join(DATA_DIR, "vertical_prioritisation_scorecard.csv"))
    granger = pd.read_csv(os.path.join(DATA_DIR, "granger_centrality.csv"))
    cold_start = pd.read_csv(os.path.join(DATA_DIR, "cold_start_nlp_benchmark_full_3.4M.csv"))
    return three_way, scorecard, granger, cold_start

@st.cache_data(show_spinner=False)
def get_json_data():
    with open(os.path.join(DATA_DIR, "lstm_results.json"), "r") as f:
        lstm = json.load(f)
    with open(os.path.join(DATA_DIR, "prophet_results.json"), "r") as f:
        prophet = json.load(f)
    with open(os.path.join(DATA_DIR, "var_results.json"), "r") as f:
        var = json.load(f)
    return lstm, prophet, var

# Load datasets
norm_df, train_df, test_df = get_time_series_data()
three_way, scorecard, granger, cold_start_summary = get_tables_data()
lstm_results, prophet_results, var_results = get_json_data()

depts = sorted([c for c in norm_df.columns if c not in ["relative_week", "Unnamed: 0"]])

# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.markdown("## 📦 SteelFlow Intelligence")
st.sidebar.caption("Big Data Demand Forecasting Platform")

views_list = [
    "1. Executive Overview & KPIs",
    "2. Timeline & Demand Normalization",
    "3. Three-Way Model Leaderboard",
    "4. LSTM Neural Network Diagnostics",
    "5. Granger Causality Demand Network",
    "6. Multivariate Cluster VAR System",
    "7. Vertical Prioritization Scorecard",
    "8. Cold-Start SKU NLP Engine",
]

view = st.sidebar.radio("Select Decision View:", views_list)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Scope:** 3,417,064 Transactions · 47,879 SKUs · 19 Verticals · Weeks 0–51")

# ------------------------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------------------------
if view.startswith("1."):
    st.title("📦 SteelFlow Logistics: Executive Demand Ecosystem")
    st.markdown("<p style='font-size:1.15rem; color:#475569;'>Production-Grade Forecasting & Network Intelligence Architecture</p>", unsafe_allow_html=True)
    st.write("")
    
    # Custom non-truncating KPI metric boxes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Transaction Volume</div>
            <div class="kpi-value">3.42M Orders</div>
            <div class="kpi-delta-green">↑ 100% Full Population</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Mean LSTM Test MAPE</div>
            <div class="kpi-value">5.43%</div>
            <div class="kpi-delta-green">↓ Leading 10 Verticals</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Granger Demand Links</div>
            <div class="kpi-value">117 Significant</div>
            <div class="kpi-delta-green">↑ p < 0.05 Causal Ties</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Catalogue Scope</div>
            <div class="kpi-value">19 Verticals</div>
            <div class="kpi-delta-blue">52 Timeline Weeks</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
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
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------------------
# VIEW 2: TIMELINE & NORMALIZATION
# ------------------------------------------------------------------------------
elif view.startswith("2."):
    st.title("📈 Timeline Reconstruction & Macro Exposure Normalization")
    st.markdown("Reconstructed user-relative purchase timeline across Weeks 0–51 to eliminate synthetic onboarding decay.")
    
    selected_dept = st.selectbox("Select Department to Inspect Normalized Series:", depts, index=depts.index("produce") if "produce" in depts else 0)
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=train_df.index, y=train_df[selected_dept], mode="lines+markers", name="Training Split (Wks 0–42)", line=dict(color="#2563EB", width=2.5)))
    fig.add_trace(go.Scattergl(x=test_df.index, y=test_df[selected_dept], mode="lines+markers", name="Evaluation Horizon (Wks 43–51)", line=dict(color="#EF4444", width=2.5)))
    fig.add_vline(x=42.5, line_dash="dash", line_color="#F59E0B", annotation_text="Train / Test Boundary")
    fig.update_layout(
        title=f"Demand per Active User: {selected_dept.upper()}",
        xaxis_title="Relative Week (0–51)",
        yaxis_title="Normalized Weekly Demand",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------------------
# VIEW 3: THREE-WAY MODEL LEADERBOARD
# ------------------------------------------------------------------------------
elif view.startswith("3."):
    st.title("🏆 Three-Way Forecasting Benchmark (Weeks 43–51 Test Split)")
    st.markdown("Head-to-head performance audit: **Auto-ARIMA vs. Facebook Prophet vs. Walk-Forward LSTM**.")
    
    st.dataframe(three_way.style.highlight_min(subset=["arima_mape", "prophet_mape", "lstm_mape"], axis=1, color="#DBEAFE"), use_container_width=True)
    
    fig = px.bar(
        three_way, x="department", y=["arima_mape", "prophet_mape", "lstm_mape"],
        barmode="group", title="MAPE (%) Comparison across 19 Departments",
        labels={"value": "MAPE (%)", "variable": "Model Architecture"},
        color_discrete_map={"arima_mape": "#10B981", "prophet_mape": "#F59E0B", "lstm_mape": "#2563EB"}
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------------------------
# VIEW 4: LSTM DIAGNOSTICS
# ------------------------------------------------------------------------------
elif view.startswith("4."):
    st.title("🧠 Walk-Forward Keras LSTM Forecaster")
    st.markdown("Out-of-sample evaluation on Weeks 43–51 using an expanding walk-forward horizon.")
    
    dept_choice = st.selectbox("Select Department:", depts, index=depts.index("frozen") if "frozen" in depts else 0)
    
    d_res = lstm_results.get(dept_choice) or lstm_results.get(dept_choice.lower()) or lstm_results.get(dept_choice.capitalize()) or {}
    actual_vals = d_res.get("actual") or d_res.get("actual_test") or d_res.get("y_true") or []
    pred_vals = d_res.get("preds") or d_res.get("pred_test") or d_res.get("predictions") or d_res.get("y_pred") or []
    mape_val = d_res.get("mape") or d_res.get("test_mape") or 0.0
    rmse_val = d_res.get("rmse") or d_res.get("test_rmse") or 0.0
    
    eval_weeks = list(range(43, 43 + len(actual_vals))) if actual_vals else list(range(43, 52))
    
    fig = go.Figure()
    if len(actual_vals) > 0:
        fig.add_trace(go.Scattergl(x=eval_weeks, y=actual_vals, mode="lines+markers", name="Actual Ground Truth", line=dict(color="#10B981", width=3)))
    if len(pred_vals) > 0:
        fig.add_trace(go.Scattergl(x=eval_weeks, y=pred_vals, mode="lines+markers", name="LSTM Walk-Forward Prediction", line=dict(color="#2563EB", width=3, dash="dash")))
        
    fig.update_layout(
        title=f"{dept_choice.upper()} — Test Forecast (MAPE: {mape_val:.2f}%, RMSE: {rmse_val:.4f})",
        xaxis_title="Evaluation Week (43–51)",
        yaxis_title="Normalized Demand per User",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Out-of-Sample MAPE", f"{mape_val:.2f}%")
    col_m2.metric("Out-of-Sample RMSE", f"{rmse_val:.5f}")

# ------------------------------------------------------------------------------
# VIEW 5: GRANGER CAUSALITY NETWORK
# ------------------------------------------------------------------------------
elif view.startswith("5."):
    st.title("🌐 Pairwise Granger Causality & Lead-Lag Network")
    st.markdown("Evaluated on differenced series ($p < 0.05$) across full 3.42M transaction history (**117 causal connections**).")
    
    fig = px.bar(
        granger.sort_values("net_leadership", ascending=False),
        x="department", y="net_leadership",
        color="net_leadership", color_continuous_scale="Blues",
        title="Net Leadership Metric (Out-Degree minus In-Degree)"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(granger, use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 6: MULTIVARIATE CLUSTER VAR
# ------------------------------------------------------------------------------
elif view.startswith("6."):
    st.title("📈 Cluster Vector Autoregression System: `Alcohol-Beverages-Snacks-Breakfast`")
    st.markdown("Capturing inter-category cross-elasticity, substitution effects, and co-movement feedback loops.")
    
    var_df = pd.DataFrame(var_results).T
    st.dataframe(var_df.style.format({"rmse": "{:.5f}", "mape": "{:.2f}%"}), use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 7: DECISION SCORECARD
# ------------------------------------------------------------------------------
elif view.startswith("7."):
    st.title("🎖️ Multi-Factor Vertical Prioritization Decision Scorecard")
    st.markdown(r"$$\text{Score} = (0.35\times\text{Vol} + 0.25\times\text{Acc} + 0.20\times\text{Stab} + 0.20\times\text{Cent}) \times \text{Reliability Multiplier}$$")
    
    st.dataframe(scorecard, use_container_width=True)

# ------------------------------------------------------------------------------
# VIEW 8: COLD-START NLP ENGINE
# ------------------------------------------------------------------------------
elif view.startswith("8."):
    st.title("🔍 TF-IDF Semantic Cold-Start SKU Forecaster")
    st.markdown("Benchmarking 4 Estimators for New Item Introductions across 3.42M orders.")
    
    st.dataframe(cold_start_summary.style.highlight_min(subset=["MAE", "RMSE", "WAPE (%)"], color="#DBEAFE"), use_container_width=True)