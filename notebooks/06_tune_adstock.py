import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_trainer import MarketingMixModel

# ==========================================
# 1. LOAD DATA & CREATE CHRONOLOGICAL SPLIT
# ==========================================
data_path = PROJECT_ROOT / "data" / "mmm_example.csv"
df = pd.read_csv(data_path)
df["date_week"] = pd.to_datetime(df["date_week"])
df = df.sort_values("date_week").reset_index(drop=True)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("=" * 60)
print("MARKETING MIX MODEL — TIME-SERIES HYPERPARAMETER TUNING")
print("=" * 60)
print(f"Total observations : {len(df)}")
print(f"Training split     : {len(train_df)} weeks (First 80%)")
print(f"Held-out test split: {len(test_df)} weeks (Final 20% — Sealed)")

# ==========================================
# 2. RUN TIMESERIES SPLIT CV ON TRAINING SET
# ==========================================
print("\nRunning 5-fold TimeSeriesSplit cross-validation on training set...")
decay_candidates = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
sat_candidates = ["none", "log", "hill", "sqrt"]

tuning_results = MarketingMixModel.cross_validate_hyperparameters(
    train_df,
    n_splits=5,
    decay_candidates=decay_candidates,
    sat_candidates=sat_candidates,
)

print("\n========== TOP HYPERPARAMETER CONFIGURATIONS (BY CV MAE) ==========")
print(tuning_results.head(10).to_string(index=False))

best_config = tuning_results.iloc[0]
print("\n" + "=" * 60)
print(f"WINNING CONFIGURATION (Selected on Train CV only):")
print(f"  Decay Rate       : {best_config['decay_rate']}")
print(f"  Saturation Curve : {best_config['saturation_curve']}")
print(f"  CV MAE           : {best_config['cv_mae']:.2f}")
print(f"  CV RMSE          : {best_config['cv_rmse']:.2f}")
print(f"  CV R²            : {best_config['cv_r2']:.4f}")
print("=" * 60)

output_path = PROJECT_ROOT / "data" / "adstock_tuning_results.csv"
tuning_results.to_csv(output_path, index=False)
print(f"\nTuning results saved to {output_path}")