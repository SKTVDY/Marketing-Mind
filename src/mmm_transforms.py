import numpy as np
import pandas as pd


def calculate_adstock(series, decay_rate: float = 0.5) -> np.ndarray:
    """
    Computes geometric adstock carryover transformation.

    Formula:
        Adstock[t] = x[t] + decay_rate * Adstock[t - 1]

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Raw marketing spend / activity values.
    decay_rate : float
        Memory decay factor in [0.0, 1.0).
    """
    if isinstance(series, pd.Series):
        values = series.to_numpy(dtype=float)
    else:
        values = np.asarray(series, dtype=float)

    if len(values) == 0:
        return np.array([], dtype=float)

    decay_rate = float(np.clip(decay_rate, 0.0, 0.999))
    adstock = np.zeros(len(values), dtype=float)
    adstock[0] = values[0]

    for i in range(1, len(values)):
        adstock[i] = values[i] + decay_rate * adstock[i - 1]

    return adstock


def apply_saturation(
    values,
    scale: float,
    curve_type: str = "log",
    hill_shape: float = 1.0,
):
    """
    Applies a non-linear saturation transformation to model diminishing marginal returns.

    Parameters
    ----------
    values : float, list, np.ndarray, or pd.Series
        Input spend or adstocked marketing values.
    scale : float
        Characteristic scale or half-saturation parameter (K). Must be > 0.
    curve_type : str
        Type of saturation curve:
        - 'log': Natural log saturation: ln(1 + x / scale)
        - 'hill': Hill / Michaelis-Menten: (x^S) / (scale^S + x^S)
        - 'sqrt': Power / square-root concave curve: sqrt(x / scale)
        - 'linear': Normalized linear scaling: x / scale
    hill_shape : float
        Slope / shape parameter for Hill function (default 1.0).

    Returns
    -------
    np.ndarray or float
        Transformed saturated values exhibiting diminishing marginal returns.
    """
    is_scalar = np.isscalar(values)
    x = np.asarray(values, dtype=float)
    x_clipped = np.clip(x, 0.0, None)

    scale = max(float(scale), 1e-8)
    curve_type = curve_type.lower()

    if curve_type == "log":
        # Strictly concave: f(0)=0, f'(x) > 0, f''(x) < 0
        saturated = np.log1p(x_clipped / scale)
    elif curve_type in ("hill", "michaelis_menten"):
        # Hill equation / Michaelis-Menten: bounded in [0, 1)
        s = max(float(hill_shape), 0.1)
        x_s = x_clipped ** s
        k_s = scale ** s
        saturated = x_s / (k_s + x_s + 1e-12)
    elif curve_type == "sqrt":
        # Standard square-root diminishing returns
        saturated = np.sqrt(x_clipped / scale)
    elif curve_type in ("linear", "none"):
        # Simple scale normalization
        saturated = x_clipped / scale
    else:
        raise ValueError(
            f"Unknown saturation curve type '{curve_type}'. "
            "Supported options are 'log', 'hill', 'sqrt', 'linear'."
        )

    return float(saturated) if is_scalar else saturated


def estimate_saturation_scale(adstock_series, method: str = "mean") -> float:
    """
    Estimates the characteristic saturation scale (K) strictly from training data
    to prevent temporal lookahead bias.

    Parameters
    ----------
    adstock_series : array-like
        Adstocked values on the training split.
    method : str
        'mean', 'median', or 'max'. Default 'mean'.
    """
    arr = np.asarray(adstock_series, dtype=float)
    positive_arr = arr[arr > 0]
    if len(positive_arr) == 0:
        return 1.0

    if method == "median":
        return float(np.median(positive_arr))
    elif method == "max":
        return float(np.max(positive_arr))
    else:
        return float(np.mean(positive_arr))
