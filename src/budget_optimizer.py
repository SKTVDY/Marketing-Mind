import numpy as np
from scipy.optimize import minimize


def optimize_budget(
    total_budget=100000,
    min_spend=10000,
    max_spend=90000,
):
    """
    Find a budget allocation between two marketing channels.

    Note: The response coefficients below are illustrative assumptions,
    not values learned from verified marketing spend and revenue data.
    """

    # Illustrative channel response parameters
    channel_1_effectiveness = 2.0
    channel_2_effectiveness = 1.5

    def predicted_return(budget):
        spend_1, spend_2 = budget

        # Diminishing returns as spending increases
        return (
            channel_1_effectiveness * np.sqrt(spend_1)
            + channel_2_effectiveness * np.sqrt(spend_2)
        )

    def objective(budget):
        # scipy minimizes, so negate the predicted return
        return -predicted_return(budget)

    constraints = [
        {
            "type": "eq",
            "fun": lambda budget: np.sum(budget) - total_budget,
        }
    ]

    bounds = [
        (min_spend, max_spend),
        (min_spend, max_spend),
    ]

    initial_budget = [
    total_budget * 16 / 25,
    total_budget * 9 / 25,
]

    result = minimize(
        objective,
        initial_budget,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    print("Optimization successful:", result.success)
    print("Optimizer message:", result.message)
    print("Raw allocation:", result.x)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    spend_1, spend_2 = result.x

    return {
        "total_budget": total_budget,
        "channel_1_budget": round(spend_1, 2),
        "channel_2_budget": round(spend_2, 2),
        "predicted_return_score": round(
            predicted_return(result.x), 2
        ),
    }


if __name__ == "__main__":
    allocation = optimize_budget()

    print("\n===== MarketingMind Budget Optimizer =====")
    print(f"Total budget: ₹{allocation['total_budget']:,.2f}")
    print(f"Channel 1 allocation: ₹{allocation['channel_1_budget']:,.2f}")
    print(f"Channel 2 allocation: ₹{allocation['channel_2_budget']:,.2f}")
    print(f"Predicted return score: {allocation['predicted_return_score']}")