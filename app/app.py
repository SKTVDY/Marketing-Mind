import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.budget_optimizer import optimize_budget
from src.model_trainer import MarketingMixModel
from src.mmm_transforms import apply_saturation

# Page configuration
st.set_page_config(
    page_title="MarketingMind | Predict & Optimize MMM",
    page_icon="📊",
    layout="wide",
)

st.title("📊 MarketingMind")
st.subheader("Predict, Then Optimize — Marketing Mix Modeling & Budget Allocation")

st.markdown(
    """
    *Empirically estimate channel response dynamics using time-series regression with **adstock carryover** 
    and **diminishing returns saturation curves**, then dynamically solve optimal budget allocations via **SciPy SLSQP**.*
    """
)

# ==========================================
# CACHED DATA & MODEL LOADING
# ==========================================
@st.cache_data
def load_data():
    data_path = PROJECT_ROOT / "data" / "mmm_example.csv"
    df = pd.read_csv(data_path)
    df["date_week"] = pd.to_datetime(df["date_week"])
    df = df.sort_values("date_week").reset_index(drop=True)
    return df


@st.cache_resource
def get_trained_model(decay_rate: float, sat_type: str):
    df = load_data()
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    model = MarketingMixModel(decay_rate=decay_rate, sat_type=sat_type)
    model.fit(train_df)
    return model


df = load_data()
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ Budget Allocation Settings")

total_budget = st.sidebar.number_input(
    "Total Marketing Budget (₹)",
    min_value=10000,
    max_value=10000000,
    value=100000,
    step=10000,
)

min_spend = st.sidebar.number_input(
    "Minimum Spend per Channel (₹)",
    min_value=0,
    max_value=int(total_budget // 2),
    value=10000,
    step=5000,
)

max_spend = st.sidebar.number_input(
    "Maximum Spend per Channel (₹)",
    min_value=int(total_budget // 2),
    max_value=int(total_budget),
    value=int(total_budget * 0.9),
    step=5000,
)

st.sidebar.divider()
st.sidebar.header("🧠 MMM Response Model Settings")

sat_curve_choice = st.sidebar.selectbox(
    "Saturation Curve (Diminishing Returns)",
    options=["log", "hill", "sqrt", "none"],
    index=0,
    format_func=lambda x: {
        "log": "Logarithmic: ln(1 + x / K)",
        "hill": "Hill / Michaelis-Menten: x / (K + x)",
        "sqrt": "Square-Root: √(x / K)",
        "none": "Linear (No Saturation)",
    }[x],
    help="Models diminishing marginal returns as channel spend increases.",
)

decay_rate_choice = st.sidebar.slider(
    "Adstock Decay Factor (λ)",
    min_value=0.0,
    max_value=0.9,
    value=0.4,
    step=0.1,
    help="Memory carryover rate: fraction of marketing impact that lingers into subsequent weeks.",
)

# Train or retrieve cached model
active_model = get_trained_model(decay_rate=decay_rate_choice, sat_type=sat_curve_choice)
channel_params = active_model.get_channel_parameters()
test_eval = active_model.evaluate(test_df)

# ==========================================
# SECTION 1: PREDICT, THEN OPTIMIZE
# ==========================================
st.header("🎯 'Predict, Then Optimize' Engine")

# Display fitted empirical response parameters
with st.container():
    st.markdown("##### 📌 Empirically Fitted Channel Parameters (Learned via Regression)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Channel 1 Weight (β₁)", f"{channel_params['beta_1']:,.1f}", help="Marginal coefficient learned from data")
    c2.metric("Channel 2 Weight (β₂)", f"{channel_params['beta_2']:,.1f}", help="Marginal coefficient learned from data")
    c3.metric("Adstock Decay (λ)", f"{channel_params['decay_rate']:.2f}")
    c4.metric("Test Set R² (Unbiased)", f"{test_eval['R2']:.3f}", help="Evaluated once on unseen 20% holdout")

# Validate spending constraints
if 2 * min_spend > total_budget:
    st.error("Error: The minimum spend across both channels exceeds total budget.")
elif 2 * max_spend < total_budget:
    st.error("Error: The maximum spend across both channels is insufficient for total budget.")
else:
    allocation = optimize_budget(
        total_budget=total_budget,
        min_spend=min_spend,
        max_spend=max_spend,
        model=active_model,
    )

    channel_1 = allocation["channel_1_budget"]
    channel_2 = allocation["channel_2_budget"]
    opt_score = allocation["predicted_return_score"]
    eq_score = allocation["equal_split_return_score"]
    improvement = allocation["improvement_pct"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Budget", f"₹{total_budget:,.0f}")
    col2.metric("Optimized Return Score", f"{opt_score:,.2f}")
    col3.metric("Equal Split Benchmark", f"{eq_score:,.2f}")
    col4.metric("Optimization Lift", f"+{improvement:.2f}%", delta=f"{improvement:.2f}%")

    st.divider()

    # Visual budget breakdown
    left, right = st.columns(2)
    with left:
        st.subheader("Recommended Budget Allocation")
        st.write(f"**Channel 1:** ₹{channel_1:,.2f} ({channel_1 / total_budget * 100:.1f}%)")
        st.progress(min(channel_1 / total_budget, 1.0))

        st.write(f"**Channel 2:** ₹{channel_2:,.2f} ({channel_2 / total_budget * 100:.1f}%)")
        st.progress(min(channel_2 / total_budget, 1.0))

        # Comparison table
        comparison = pd.DataFrame(
            {
                "Strategy": ["Equal Split (50/50)", "SLSQP Optimized"],
                "Channel 1 (₹)": [total_budget / 2.0, channel_1],
                "Channel 2 (₹)": [total_budget / 2.0, channel_2],
                "Channel 1 %": ["50.0%", f"{channel_1 / total_budget * 100:.1f}%"],
                "Channel 2 %": ["50.0%", f"{channel_2 / total_budget * 100:.1f}%"],
                "Predicted Return": [eq_score, opt_score],
            }
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        # Download button
        csv_data = comparison.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Allocation as CSV",
            data=csv_data,
            file_name="marketingmind_optimal_allocation.csv",
            mime="text/csv",
        )

    with right:
        st.subheader("📈 Diminishing Returns Response Curves")
        # Plot continuous response curves and show optimal allocation points
        spend_range = np.linspace(0, total_budget, 200)
        norm_spend = spend_range / total_budget

        resp_ch1 = channel_params["beta_1"] * apply_saturation(
            norm_spend, scale=channel_params["scale_1"], curve_type=channel_params["sat_type"]
        )
        resp_ch2 = channel_params["beta_2"] * apply_saturation(
            norm_spend, scale=channel_params["scale_2"], curve_type=channel_params["sat_type"]
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(spend_range, resp_ch1, label=f"Channel 1 (β₁={channel_params['beta_1']:.0f})", color="#1f77b4", lw=2)
        ax.plot(spend_range, resp_ch2, label=f"Channel 2 (β₂={channel_params['beta_2']:.0f})", color="#ff7f0e", lw=2)

        # Mark optimal allocation points
        opt_resp_1 = channel_params["beta_1"] * apply_saturation(
            channel_1 / total_budget, scale=channel_params["scale_1"], curve_type=channel_params["sat_type"]
        )
        opt_resp_2 = channel_params["beta_2"] * apply_saturation(
            channel_2 / total_budget, scale=channel_params["scale_2"], curve_type=channel_params["sat_type"]
        )

        ax.scatter([channel_1], [opt_resp_1], color="#1f77b4", s=100, zorder=5, label=f"Ch 1 Opt (₹{channel_1:,.0f})")
        ax.scatter([channel_2], [opt_resp_2], color="#ff7f0e", s=100, zorder=5, label=f"Ch 2 Opt (₹{channel_2:,.0f})")

        ax.set_xlabel("Spend (₹)")
        ax.set_ylabel("Incremental Return Contribution")
        ax.set_title(f"Channel Response Under {sat_curve_choice.upper()} Saturation")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==========================================
# SECTION 2: TABS FOR DEEP ANALYSIS
# ==========================================
st.divider()
tab1, tab2, tab3 = st.tabs([
    "🧪 Model Validation & TimeSeriesSplit CV",
    "📊 Marketing Data Analytics (EDA)",
    "🔮 Scenario Forecaster",
])

# -------------------------------------------------------------
# TAB 1: VALIDATION & CV
# -------------------------------------------------------------
with tab1:
    st.subheader("Rigorous Time-Series Validation Pipeline")
    st.markdown(
        """
        - **Data Leakage Prevention:** Saturation scales and adstock transformations are fit **strictly on training observations**.
        - **Hyperparameter Tuning:** Tuned via **5-fold TimeSeriesSplit cross-validation** within the training window (no future lookahead).
        - **Unbiased Assessment:** The held-out test split (final 20%) is evaluated **once** to report true generalization metrics.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🏆 Unbiased Held-Out Test Performance")
        val_csv_path = PROJECT_ROOT / "data" / "adstock_validation_results.csv"
        if val_csv_path.exists():
            val_df = pd.read_csv(val_csv_path)
            st.dataframe(val_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run notebooks/07_validate_adstock.py to generate final validation comparison.")

    with col2:
        st.markdown("##### 🔄 TimeSeriesSplit Cross-Validation Results")
        tune_csv_path = PROJECT_ROOT / "data" / "adstock_tuning_results.csv"
        if tune_csv_path.exists():
            tune_df = pd.read_csv(tune_csv_path)
            st.dataframe(tune_df.head(6), use_container_width=True, hide_index=True)
        else:
            st.info("Run notebooks/06_tune_adstock.py to generate tuning results.")

    # Time series prediction chart
    st.markdown("##### 📅 Test Period Forecast (Actual vs Predicted)")
    test_preds = active_model.predict(test_df)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_df["date_week"], test_df["y"], marker="o", label="Actual Outcome", color="#2ca02c")
    ax.plot(test_df["date_week"], test_preds, marker="x", linestyle="--", label="Model Predicted", color="#d62728")
    ax.set_xlabel("Week")
    ax.set_ylabel("Target Metric (y)")
    ax.set_title("Out-of-Sample Holdout Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS
# -------------------------------------------------------------
with tab2:
    st.subheader("Historical Marketing Data Exploration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Weeks", f"{len(df):,}")
    c2.metric("Date Range", f"{df['date_week'].dt.strftime('%b %Y').iloc[0]} – {df['date_week'].dt.strftime('%b %Y').iloc[-1]}")
    c3.metric("Average Outcome", f"{df['y'].mean():,.1f}")
    c4.metric("Missing Values", f"{df.isnull().sum().sum()}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### Outcome Trend Over Time")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["date_week"], df["y"], color="#1f77b4", lw=1.5)
        ax.axvline(train_df["date_week"].iloc[-1], color="red", linestyle="--", label="80% Train/Test Split")
        ax.set_xlabel("Week")
        ax.set_ylabel("Outcome (y)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_r:
        st.markdown("##### Correlation Matrix")
        num_cols = df.select_dtypes(include="number").columns
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("📄 View Underlying Dataset"):
        st.dataframe(df, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: SCENARIO FORECASTER
# -------------------------------------------------------------
with tab3:
    st.subheader("Weekly Target Metric Forecaster")
    st.write("Simulate expected weekly outcome given specific marketing spend levels and calendar events.")

    latest = df.iloc[-1]
    with st.form("scenario_form"):
        f1, f2 = st.columns(2)
        with f1:
            input_x1 = st.slider("Channel 1 Activity Index (x1)", 0.0, 1.5, float(latest["x1"]), 0.05)
            input_x2 = st.slider("Channel 2 Activity Index (x2)", 0.0, 1.5, float(latest["x2"]), 0.05)
            input_ev1 = st.selectbox("Event 1 Active?", [0, 1], index=int(latest["event_1"]))
        with f2:
            input_ev2 = st.selectbox("Event 2 Active?", [0, 1], index=int(latest["event_2"]))
            input_doy = st.number_input("Day of Year", 1, 366, int(latest["dayofyear"]))
            input_t = st.number_input("Time Trend Index (t)", 0, 300, int(latest["t"]) + 1)

        predict_btn = st.form_submit_button("Forecast Expected Outcome")

    if predict_btn:
        # Create single-row scenario dataframe
        sim_df = pd.DataFrame([{
            "x1": input_x1,
            "x2": input_x2,
            "event_1": input_ev1,
            "event_2": input_ev2,
            "dayofyear": input_doy,
            "t": input_t,
        }])
        sim_pred = active_model.predict(sim_df)[0]
        st.success(f"🎯 **Forecasted Weekly Target Outcome:** {sim_pred:,.2f}")
        st.caption("Forecast combines baseline effects, calendar seasonality, and saturated adstock channel returns.")