import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mmm_transforms import apply_saturation

# Global cached default model for fast repeated calls
_DEFAULT_MODEL = None


def get_default_model(data_path: Optional[Union[str, Path]] = None):
    """
    Loads or trains the default MarketingMixModel on the dataset
    using the best validated hyperparameters.
    """
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is not None:
        return _DEFAULT_MODEL

    from src.model_trainer import MarketingMixModel

    if data_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_path = base_dir / "data" / "mmm_example.csv"

    data_path = Path(data_path)
    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)
    df["date_week"] = pd.to_datetime(df["date_week"])
    df = df.sort_values("date_week").reset_index(drop=True)

    # Train strictly on the 80% training split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()

    # Optimal parameters identified through TimeSeriesSplit CV
    model = MarketingMixModel(decay_rate=0.4, sat_type="log")
    model.fit(train_df)
    _DEFAULT_MODEL = model
    return _DEFAULT_MODEL


def optimize_budget(
    total_budget: float = 100000.0,
    min_spend: float = 10000.0,
    max_spend: float = 90000.0,
    model: Optional[Any] = None,
    channel_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Finds the optimal budget allocation across marketing channels using SciPy SLSQP.

    Connects the optimizer directly to the Marketing Mix Model (MMM) regression
    coefficients and saturation curves, replacing hardcoded parameters with
    empirically fitted response dynamics ('Predict, then Optimize').

    Parameters
    ----------
    total_budget : float
        Total budget to distribute across channels (e.g. ₹100,000).
    min_spend : float
        Minimum budget allowed for each individual channel.
    max_spend : float
        Maximum budget allowed for each individual channel.
    model : Optional[MarketingMixModel]
        A fitted MarketingMixModel instance. If None, default model is loaded.
    channel_params : Optional[Dict[str, Any]]
        Explicit channel parameter dictionary with keys:
        'beta_1', 'beta_2', 'scale_1', 'scale_2', 'sat_type'.

    Returns
    -------
    dict
        Optimal channel allocations, predicted return scores, equal split baseline,
        and percentage improvement.
    """
    total_budget = float(total_budget)
    min_spend = float(min_spend)
    max_spend = float(max_spend)

    # 1. Resolve channel response parameters from fitted model
    if channel_params is not None:
        params = channel_params
    elif model is not None:
        params = model.get_channel_parameters()
    else:
        loaded_model = get_default_model()
        if loaded_model is not None:
            params = loaded_model.get_channel_parameters()
        else:
            # Fallback if dataset is not found in path
            params = {
                "beta_1": 2100.0,
                "beta_2": 1210.0,
                "scale_1": 0.35,
                "scale_2": 0.20,
                "sat_type": "log",
                "decay_rate": 0.4,
                "intercept": 2640.0,
                "baseline_outcome": 5200.0,
            }

    beta_1 = float(params.get("beta_1", 2100.0))
    beta_2 = float(params.get("beta_2", 1210.0))
    scale_1 = float(params.get("scale_1", 0.35))
    scale_2 = float(params.get("scale_2", 0.20))
    sat_type = str(params.get("sat_type", "log"))
    baseline_outcome = float(params.get("baseline_outcome", 0.0))

    # Reference spend scale to map currency spend (e.g. ₹100k) into model input space
    spend_scale = total_budget if total_budget > 0 else 100000.0

    def predicted_return(budget_vec: np.ndarray) -> float:
        """
        Computes response using the model's fitted coefficients and saturation curve.
        """
        spend_1, spend_2 = budget_vec
        s1 = spend_1 / spend_scale
        s2 = spend_2 / spend_scale

        sat_1 = apply_saturation(s1, scale=scale_1, curve_type=sat_type)
        sat_2 = apply_saturation(s2, scale=scale_2, curve_type=sat_type)

        return beta_1 * sat_1 + beta_2 * sat_2

    def objective(budget_vec: np.ndarray) -> float:
        # SciPy minimizes, so negate return
        return -predicted_return(budget_vec)

    # Constraint: sum(budget) == total_budget
    constraints = [
        {
            "type": "eq",
            "fun": lambda b: np.sum(b) - total_budget,
        }
    ]

    bounds = [
        (min_spend, max_spend),
        (min_spend, max_spend),
    ]

    # Initial guess: proportional to beta coefficients
    coef_sum = (beta_1 + beta_2) if (beta_1 + beta_2) > 0 else 2.0
    initial_budget = np.array(
        [
            total_budget * (beta_1 / coef_sum),
            total_budget * (beta_2 / coef_sum),
        ]
    )
    # Clip initial budget to valid bounds
    initial_budget = np.clip(initial_budget, min_spend, max_spend)
    # Rebalance to ensure exact equality constraint
    initial_budget[1] = total_budget - initial_budget[0]

    result = minimize(
        objective,
        initial_budget,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200},
    )

    if not result.success:
        # Fallback to bounded initial allocation if optimizer encounters an edge case
        spend_1 = float(np.clip(initial_budget[0], min_spend, max_spend))
        spend_2 = total_budget - spend_1
    else:
        spend_1, spend_2 = float(result.x[0]), float(result.x[1])

    optimized_return_score = predicted_return(np.array([spend_1, spend_2]))

    # Compare with equal split benchmark
    equal_budget = np.array([total_budget / 2.0, total_budget / 2.0])
    equal_return_score = predicted_return(equal_budget)

    improvement_pct = (
        ((optimized_return_score - equal_return_score) / equal_return_score * 100.0)
        if equal_return_score > 0
        else 0.0
    )

    return {
        "total_budget": round(total_budget, 2),
        "channel_1_budget": round(spend_1, 2),
        "channel_2_budget": round(spend_2, 2),
        "predicted_return_score": round(optimized_return_score, 2),
        "equal_split_return_score": round(equal_return_score, 2),
        "improvement_pct": round(improvement_pct, 2),
        "total_predicted_outcome": round(optimized_return_score + baseline_outcome, 2),
        "model_parameters": {
            "beta_1": round(beta_1, 2),
            "beta_2": round(beta_2, 2),
            "scale_1": round(scale_1, 4),
            "scale_2": round(scale_2, 4),
            "saturation_curve": sat_type,
            "decay_rate": params.get("decay_rate", 0.4),
        },
    }


if __name__ == "__main__":
    allocation = optimize_budget()

    print("\n===== MarketingMind 'Predict, Then Optimize' Engine =====")
    print(f"Total budget: INR {allocation['total_budget']:,.2f}")
    print(f"Channel 1 allocation: INR {allocation['channel_1_budget']:,.2f}")
    print(f"Channel 2 allocation: INR {allocation['channel_2_budget']:,.2f}")
    print(f"Optimized return score: {allocation['predicted_return_score']:,.2f}")
    print(f"Equal split score: {allocation['equal_split_return_score']:,.2f}")
    print(f"Improvement vs Equal Split: +{allocation['improvement_pct']:.2f}%")
    print(f"Model parameters used: {allocation['model_parameters']}")