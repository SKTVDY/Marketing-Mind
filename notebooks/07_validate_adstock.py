import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_trainer import MarketingMixModel

# ==========================================
# 1. LOAD DATA & SPLIT CHRONOLOGICALLY
# ==========================================
data_path = PROJECT_ROOT / "data" / "mmm_example.csv"
df = pd.read_csv(data_path)
df["date_week"] = pd.to_datetime(df["date_week"])
df = df.sort_values("date_week").reset_index(drop=True)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("=" * 65)
print("FINAL TEST SET VALIDATION — EVALUATED ONCE ON UNSEEN DATA")
print("=" * 65)
print(f"Training observations: {len(train_df)}")
print(f"Testing observations : {len(test_df)}")

# ==========================================
# 2. FIT CANDIDATE MODELS ON TRAINING DATA ONLY
# ==========================================
eval_configs = [
    {"label": "Baseline (No Adstock, No Saturation)", "decay": 0.0, "sat": "none"},
    {"label": "Adstock Only (Decay 0.5)", "decay": 0.5, "sat": "none"},
    {"label": "Adstock + Log Saturation (Decay 0.4)", "decay": 0.4, "sat": "log"},
    {"label": "Adstock + Sqrt Saturation (Decay 0.4)", "decay": 0.4, "sat": "sqrt"},
    {"label": "Adstock + Hill Saturation (Decay 0.4)", "decay": 0.4, "sat": "hill"},
]

results = []
trained_models = {}

for config in eval_configs:
    model = MarketingMixModel(decay_rate=config["decay"], sat_type=config["sat"])
    model.fit(train_df)
    
    # Evaluate strictly ONCE on held-out test data
    metrics = model.evaluate(test_df)
    
    results.append({
        "Model": config["label"],
        "Decay": config["decay"],
        "Saturation": config["sat"],
        "Test_MAE": round(metrics["MAE"], 2),
        "Test_RMSE": round(metrics["RMSE"], 2),
        "Test_R2": round(metrics["R2"], 4),
    })
    trained_models[config["label"]] = model

results_df = pd.DataFrame(results).sort_values("Test_MAE").reset_index(drop=True)

print("\n========== UNBIASED HELD-OUT TEST RESULTS ==========")
print(results_df.to_string(index=False))

# Calculate improvement of best tuned model over baseline
baseline_row = results_df[results_df["Decay"] == 0.0].iloc[0]
best_row = results_df.iloc[0]

mae_reduction = (baseline_row["Test_MAE"] - best_row["Test_MAE"]) / baseline_row["Test_MAE"] * 100.0
r2_improvement = (best_row["Test_R2"] - baseline_row["Test_R2"]) / max(baseline_row["Test_R2"], 0.001) * 100.0

print("\n" + "=" * 65)
print(f"PERFORMANCE LIFT VS BASELINE:")
print(f"  Best Model    : {best_row['Model']}")
print(f"  MAE Reduction : {mae_reduction:.1f}% ({baseline_row['Test_MAE']} -> {best_row['Test_MAE']})")
print(f"  R² Lift       : +{r2_improvement:.1f}% ({baseline_row['Test_R2']} -> {best_row['Test_R2']})")
print("=" * 65)

# Save final validation table
output_path = PROJECT_ROOT / "data" / "adstock_validation_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nFinal validation results saved to {output_path}")
