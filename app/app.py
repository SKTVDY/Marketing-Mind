import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Allow importing the optimizer from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.budget_optimizer import optimize_budget


# Page configuration
st.set_page_config(
    page_title="MarketingMind | Budget Optimizer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 MarketingMind")
st.subheader("AI-Powered Marketing Budget Optimizer")

st.write(
    "Explore how different budget allocations affect an illustrative "
    "marketing response model."
)

st.warning(
    "Prototype notice: Channel effectiveness values are hypothetical. "
    "Predicted return scores are not actual revenue or ROI."
)

# Sidebar inputs
st.sidebar.header("Budget Settings")

total_budget = st.sidebar.number_input(
    "Total Marketing Budget (₹)",
    min_value=20000,
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

# Validate spending constraints
if 2 * min_spend > total_budget:
    st.error("The minimum spend across both channels exceeds the total budget.")

elif 2 * max_spend < total_budget:
    st.error("The maximum spend across both channels is insufficient for the budget.")

else:
    # Recalculate the allocation with user-selected constraints
    allocation = optimize_budget(
        total_budget=total_budget,
        min_spend=min_spend,
        max_spend=max_spend,
    )

    channel_1 = allocation["channel_1_budget"]
    channel_2 = allocation["channel_2_budget"]
    optimized_score = allocation["predicted_return_score"]

    # Equal allocation comparison
    equal_budget = [total_budget / 2, total_budget / 2]

    equal_score = (
        2.0 * np.sqrt(equal_budget[0])
        + 1.5 * np.sqrt(equal_budget[1])
    )

    improvement = (
        (optimized_score - equal_score) / equal_score * 100
        if equal_score != 0
        else 0
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Budget",
        f"₹{total_budget:,.0f}",
    )

    col2.metric(
        "Optimized Return Score",
        f"{optimized_score:,.2f}",
    )

    col3.metric(
        "Improvement vs Equal Split",
        f"{improvement:.2f}%",
    )

    st.divider()

    # Allocation results
    st.header("Recommended Budget Allocation")

    left, right = st.columns(2)

    with left:
        st.metric(
            "Channel 1 Budget",
            f"₹{channel_1:,.2f}",
        )

        st.progress(
            min(channel_1 / total_budget, 1.0)
        )

    with right:
        st.metric(
            "Channel 2 Budget",
            f"₹{channel_2:,.2f}",
        )

        st.progress(
            min(channel_2 / total_budget, 1.0)
        )

    # Allocation chart
    chart_data = pd.DataFrame(
        {
            "Channel": ["Channel 1", "Channel 2"],
            "Budget (₹)": [channel_1, channel_2],
        }
    )

    st.subheader("Budget Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        chart_data["Channel"],
        chart_data["Budget (₹)"],
    )

    ax.set_ylabel("Budget (₹)")
    ax.set_title("Optimized Marketing Budget")
    ax.ticklabel_format(axis="y", style="plain")

    st.pyplot(fig)
    plt.close(fig)

    # Comparison table
    st.subheader("Allocation Comparison")

    comparison = pd.DataFrame(
        {
            "Strategy": ["Equal Split", "Optimized"],
            "Channel 1 (₹)": [
                equal_budget[0],
                channel_1,
            ],
            "Channel 2 (₹)": [
                equal_budget[1],
                channel_2,
            ],
            "Return Score": [
                round(equal_score, 2),
                optimized_score,
            ],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    # Downloadable allocation
    csv_data = comparison.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Allocation as CSV",
        data=csv_data,
        file_name="marketingmind_allocation.csv",
        mime="text/csv",
    )

st.divider()

st.caption(
    "MarketingMind | Budget optimization prototype | "
    "Illustrative assumptions, not verified marketing ROI."
)

# ==========================================
# DATA ANALYTICS SECTION
# ==========================================

st.divider()

st.header("📈 Marketing Data Analytics")

st.write(
    "Explore historical patterns and relationships in the supplied "
    "synthetic dataset."
)

# Load the dataset
data_path = PROJECT_ROOT / "data" / "mmm_example.csv"

try:
    df = pd.read_csv(data_path)

    # Convert dates and sort chronologically
    df["date_week"] = pd.to_datetime(df["date_week"])
    df = df.sort_values("date_week")

    # Dataset overview
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Observations", f"{len(df):,}")
    col2.metric("Variables", f"{len(df.columns):,}")
    col3.metric("Average Target", f"{df['y'].mean():,.2f}")
    col4.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    # Target metric over time
    st.subheader("Target Metric Over Time")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df["date_week"],
        df["y"],
        marker="o",
        markersize=3,
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Target metric (y)")
    ax.set_title("Weekly Target Metric Trend")

    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Relationships between variables
    st.subheader("Channel Variable Analysis")

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(df["x1"], df["y"], alpha=0.7)

        ax.set_xlabel("x1")
        ax.set_ylabel("Target metric (y)")
        ax.set_title("x1 vs Target Metric")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(df["x2"], df["y"], alpha=0.7)

        ax.set_xlabel("x2")
        ax.set_ylabel("Target metric (y)")
        ax.set_title("x2 vs Target Metric")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Correlation analysis
    st.subheader("Correlation Matrix")

    numeric_columns = df.select_dtypes(include="number").columns
    correlation = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(9, 6))

    image = ax.imshow(correlation, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index)

    for i in range(len(correlation.index)):
        for j in range(len(correlation.columns)):
            ax.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title("Correlation Between Numeric Variables")
    fig.colorbar(image, ax=ax, label="Correlation")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Display underlying dataset
    with st.expander("View Dataset"):
        st.dataframe(df, use_container_width=True)

    # Download dataset
    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="marketingmind_dataset.csv",
        mime="text/csv",
    )

except FileNotFoundError:
    st.error(
        "Dataset not found. Please check that "
        "data/mmm_example.csv exists in your project."
    )

except (KeyError, ValueError) as error:
    st.error(f"Could not process the dataset: {error}")


# ==========================================
# MACHINE LEARNING PREDICTION
# ==========================================

st.divider()

st.header("🤖 Target Metric Prediction")

st.write(
    "Train a linear regression model on historical observations "
    "and estimate the target metric for selected input values."
)

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Define model features and target
    features = [
        "x1",
        "x2",
        "event_1",
        "event_2",
        "dayofyear",
        "t",
    ]

    model_df = df.sort_values("date_week").reset_index(drop=True)

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index]
    test_df = model_df.iloc[split_index:]

    X_train = train_df[features]
    y_train = train_df["y"]

    X_test = test_df[features]
    y_test = test_df["y"]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on the future holdout period
    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)

    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("R²", f"{r2:.3f}")

    st.caption(
        f"Training observations: {len(train_df)} | "
        f"Testing observations: {len(test_df)}"
    )

    st.info(
        "The evaluation metrics describe performance on the held-out "
        "period. This model learns statistical associations, not proven "
        "causal marketing effects."
    )

    # Prediction inputs
    st.subheader("Predict the Target Metric")

    latest = model_df.iloc[-1]

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            x1_value = st.number_input(
                "x1 value",
                value=float(latest["x1"]),
                format="%.4f",
            )

            x2_value = st.number_input(
                "x2 value",
                value=float(latest["x2"]),
                format="%.4f",
            )

            event_1_value = st.selectbox(
                "event_1",
                options=[0, 1],
                index=int(latest["event_1"]),
            )

        with col2:
            event_2_value = st.selectbox(
                "event_2",
                options=[0, 1],
                index=int(latest["event_2"]),
            )

            dayofyear_value = st.number_input(
                "Day of year",
                min_value=1,
                max_value=366,
                value=int(latest["dayofyear"]),
            )

            t_value = st.number_input(
                "Time index (t)",
                min_value=0,
                value=int(latest["t"]) + 1,
            )

        submitted = st.form_submit_button("Predict Target Metric")

    if submitted:
        input_data = pd.DataFrame(
            [[
                x1_value,
                x2_value,
                event_1_value,
                event_2_value,
                dayofyear_value,
                t_value,
            ]],
            columns=features,
        )

        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted target metric: {prediction:,.2f}"
        )

        st.caption(
            "This is a model estimate, not a guaranteed outcome. "
            "Predictions outside the training data's observed ranges "
            "may be unreliable."
        )

except Exception as error:
    st.error(f"Could not run the prediction model: {error}")