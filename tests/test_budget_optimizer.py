import pytest

from src.budget_optimizer import optimize_budget


def test_default_budget_is_allocated_correctly():
    result = optimize_budget()

    assert result["total_budget"] == 100000
    assert (
        result["channel_1_budget"] + result["channel_2_budget"]
        == result["total_budget"]
    )


def test_budget_respects_minimum_and_maximum():
    result = optimize_budget(
        total_budget=100000,
        min_spend=10000,
        max_spend=90000,
    )

    assert 10000 <= result["channel_1_budget"] <= 90000
    assert 10000 <= result["channel_2_budget"] <= 90000


def test_custom_total_budget():
    result = optimize_budget(
        total_budget=50000,
        min_spend=5000,
        max_spend=45000,
    )

    assert result["total_budget"] == 50000
    assert (
        result["channel_1_budget"] + result["channel_2_budget"]
        == result["total_budget"]
    )


def test_predicted_return_score_is_positive():
    result = optimize_budget()

    assert result["predicted_return_score"] > 0