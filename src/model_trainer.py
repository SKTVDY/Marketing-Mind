import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mmm_transforms import (
    apply_saturation,
    calculate_adstock,
    estimate_saturation_scale,
)


class MarketingMixModel:
    """
    Marketing Mix Model (MMM) that estimates carryover effects (adstock),
    diminishing marginal returns (saturation), and baseline controls
    using time-series cross-validation.
    """

    def __init__(
        self,
        decay_rate: float = 0.4,
        sat_type: str = "log",
        channel_cols: Tuple[str, str] = ("x1", "x2"),
        control_cols: Tuple[str, ...] = ("event_1", "event_2", "dayofyear", "t"),
        target_col: str = "y",
    ):
        self.decay_rate = decay_rate
        self.sat_type = sat_type
        self.channel_cols = channel_cols
        self.control_cols = control_cols
        self.target_col = target_col

        self.model = LinearRegression(positive=True)
        self.channel_scales: Dict[str, float] = {}
        self.coefficients: Dict[str, float] = {}
        self.intercept: float = 0.0
        self.is_fitted: bool = False
        self.training_metrics: Dict[str, float] = {}
        self.baseline_prediction: float = 0.0

    def _transform_features(
        self,
        df: pd.DataFrame,
        fit_scales: bool = False,
    ) -> pd.DataFrame:
        """
        Transforms raw channel inputs with adstock and saturation,
        ensuring saturation scales are fit only on training data.
        """
        ch1, ch2 = self.channel_cols

        # Adstock transformation
        ch1_adstock = calculate_adstock(df[ch1], self.decay_rate)
        ch2_adstock = calculate_adstock(df[ch2], self.decay_rate)

        # Estimate saturation scales if fitting
        if fit_scales or ch1 not in self.channel_scales:
            self.channel_scales[ch1] = estimate_saturation_scale(ch1_adstock, method="mean")
        if fit_scales or ch2 not in self.channel_scales:
            self.channel_scales[ch2] = estimate_saturation_scale(ch2_adstock, method="mean")

        # Non-linear diminishing returns transformation
        ch1_saturated = apply_saturation(
            ch1_adstock,
            scale=self.channel_scales[ch1],
            curve_type=self.sat_type,
        )
        ch2_saturated = apply_saturation(
            ch2_adstock,
            scale=self.channel_scales[ch2],
            curve_type=self.sat_type,
        )

        transformed = pd.DataFrame(
            {
                f"{ch1}_saturated": ch1_saturated,
                f"{ch2}_saturated": ch2_saturated,
            },
            index=df.index,
        )

        for col in self.control_cols:
            if col in df.columns:
                transformed[col] = df[col].values

        return transformed

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the MMM regression model on the training set.
        """
        train_sorted = train_df.sort_values("date_week" if "date_week" in train_df else train_df.index.name or 0)
        X_train = self._transform_features(train_sorted, fit_scales=True)
        y_train = train_sorted[self.target_col]

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        feature_names = list(X_train.columns)
        self.coefficients = dict(zip(feature_names, self.model.coef_))
        self.intercept = float(self.model.intercept_)

        train_preds = self.model.predict(X_train)
        self.training_metrics = {
            "MAE": float(mean_absolute_error(y_train, train_preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_train, train_preds))),
            "R2": float(r2_score(y_train, train_preds)),
        }

        # Calculate average baseline outcome (intercept + average control impact)
        ch1_feat = f"{self.channel_cols[0]}_saturated"
        ch2_feat = f"{self.channel_cols[1]}_saturated"
        control_contrib = 0.0
        for col in self.control_cols:
            if col in self.coefficients:
                control_contrib += self.coefficients[col] * float(train_sorted[col].mean())

        self.baseline_prediction = self.intercept + control_contrib
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for a dataframe.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict() is called.")
        X = self._transform_features(df, fit_scales=False)
        return self.model.predict(X)

    def evaluate(self, df_eval: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluates the model against actual target outcomes.
        """
        y_true = df_eval[self.target_col].to_numpy()
        y_pred = self.predict(df_eval)

        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
        }

    def get_channel_parameters(self) -> Dict[str, Union[float, str]]:
        """
        Returns fitted channel effectiveness parameters, saturation scales,
        and decay factors for use in optimization.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        ch1, ch2 = self.channel_cols
        return {
            "channel_1_col": ch1,
            "channel_2_col": ch2,
            "beta_1": float(self.coefficients.get(f"{ch1}_saturated", 0.0)),
            "beta_2": float(self.coefficients.get(f"{ch2}_saturated", 0.0)),
            "scale_1": float(self.channel_scales.get(ch1, 1.0)),
            "scale_2": float(self.channel_scales.get(ch2, 1.0)),
            "decay_rate": float(self.decay_rate),
            "sat_type": self.sat_type,
            "intercept": float(self.intercept),
            "baseline_outcome": float(self.baseline_prediction),
        }

    def predict_channel_response(
        self,
        spend_1: float,
        spend_2: float,
        spend_scale: float = 100000.0,
        include_baseline: bool = False,
    ) -> float:
        """
        Calculates the model's predicted incremental or total outcome
        for a given budget allocation across Channel 1 and Channel 2.

        Formula:
            Return = beta_1 * Sat(s1) + beta_2 * Sat(s2) [+ Baseline]
        """
        params = self.get_channel_parameters()
        s1 = max(float(spend_1), 0.0) / max(float(spend_scale), 1.0)
        s2 = max(float(spend_2), 0.0) / max(float(spend_scale), 1.0)

        sat1 = apply_saturation(s1, scale=params["scale_1"], curve_type=params["sat_type"])
        sat2 = apply_saturation(s2, scale=params["scale_2"], curve_type=params["sat_type"])

        incremental_return = params["beta_1"] * sat1 + params["beta_2"] * sat2
        if include_baseline:
            return incremental_return + params["baseline_outcome"]
        return incremental_return

    @classmethod
    def cross_validate_hyperparameters(
        cls,
        train_df: pd.DataFrame,
        n_splits: int = 5,
        decay_candidates: Optional[List[float]] = None,
        sat_candidates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Performs TimeSeriesSplit cross-validation strictly on the training set
        to tune adstock decay rates and saturation curves without lookahead bias.
        """
        if decay_candidates is None:
            decay_candidates = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
        if sat_candidates is None:
            sat_candidates = ["none", "log", "hill", "sqrt"]

        df_sorted = train_df.sort_values(
            "date_week" if "date_week" in train_df else train_df.index.name or 0
        ).reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for sat_type in sat_candidates:
            for decay in decay_candidates:
                split_maes, split_rmses, split_r2s = [], [], []

                for tr_idx, val_idx in tscv.split(df_sorted):
                    cv_tr = df_sorted.iloc[tr_idx].copy()
                    cv_val = df_sorted.iloc[val_idx].copy()

                    cv_model = cls(decay_rate=decay, sat_type=sat_type)
                    cv_model.fit(cv_tr)

                    eval_metrics = cv_model.evaluate(cv_val)
                    split_maes.append(eval_metrics["MAE"])
                    split_rmses.append(eval_metrics["RMSE"])
                    split_r2s.append(eval_metrics["R2"])

                results.append(
                    {
                        "decay_rate": decay,
                        "saturation_curve": sat_type,
                        "cv_mae": float(np.mean(split_maes)),
                        "cv_rmse": float(np.mean(split_rmses)),
                        "cv_r2": float(np.mean(split_r2s)),
                    }
                )

        res_df = pd.DataFrame(results).sort_values("cv_mae").reset_index(drop=True)
        return res_df
