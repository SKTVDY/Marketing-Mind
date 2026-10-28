import pytest
import numpy as np
import pandas as pd

from src.mmm_transforms import (
    calculate_adstock,
    apply_saturation,
    estimate_saturation_scale,
)
from src.model_trainer import MarketingMixModel
from src.budget_optimizer import optimize_budget


def test_adstock_zero_decay():
    series = np.array([10.0, 20.0, 30.0])
    adstock = calculate_adstock(series, decay_rate=0.0)
    np.testing.assert_array_almost_equal(adstock, series)


def test_adstock_carryover_decay():
    series = np.array([10.0, 0.0, 0.0])
    adstock = calculate_adstock(series, decay_rate=0.5)
    expected = np.array([10.0, 5.0, 2.5])
    np.testing.assert_array_almost_equal(adstock, expected)


def test_saturation_diminishing_marginal_returns():
    scale = 10.0
    for curve in ["log", "hill", "sqrt"]:
        # Increments of spend
        s0 = apply_saturation(0.0, scale=scale, curve_type=curve)
        s1 = apply_saturation(10.0, scale=scale, curve_type=curve)
        s2 = apply_saturation(20.0, scale=scale, curve_type=curve)

        first_increment = s1 - s0
        second_increment = s2 - s1

        # Marginal return of first 10 spend must be strictly greater than second 10 spend
        assert first_increment > second_increment, f"Failed diminishing returns check for {curve}"


def test_model_training_and_positive_coefficients():
    # Synthetic small dataset
    n = 30
    df = pd.DataFrame({
        "date_week": pd.date_range("2023-01-01", periods=n, freq="W"),
        "x1": np.linspace(0.1, 0.9, n),
        "x2": np.linspace(0.05, 0.5, n),
        "event_1": np.zeros(n),
        "event_2": np.zeros(n),
        "dayofyear": np.arange(n) * 7 + 1,
        "t": np.arange(n),
        "y": 1000 + 500 * np.linspace(0.1, 0.9, n) + 300 * np.linspace(0.05, 0.5, n),
    })

    model = MarketingMixModel(decay_rate=0.4, sat_type="log")
    model.fit(df)

    assert model.is_fitted
    params = model.get_channel_parameters()

    # Non-negative coefficients constraint
    assert params["beta_1"] >= 0.0
    assert params["beta_2"] >= 0.0
    assert params["scale_1"] > 0.0
    assert params["scale_2"] > 0.0

    preds = model.predict(df)
    assert len(preds) == n


def test_predict_then_optimize_integration():
    allocation = optimize_budget(
        total_budget=80000,
        min_spend=10000,
        max_spend=70000,
    )

    assert allocation["total_budget"] == 80000
    assert abs(allocation["channel_1_budget"] + allocation["channel_2_budget"] - 80000) < 0.01
    assert 10000 <= allocation["channel_1_budget"] <= 70000
    assert 10000 <= allocation["channel_2_budget"] <= 70000
    assert allocation["predicted_return_score"] >= allocation["equal_split_return_score"]
    assert "model_parameters" in allocation
