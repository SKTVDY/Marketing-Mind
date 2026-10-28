import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_trainer import MarketingMixModel

# ==========================================
# 1. LOAD DATA
# ==========================================
data_path = PROJECT_ROOT / "data" / "mmm_example.csv"
df = pd.read_csv(data_path)
df["date_week"] = pd.to_datetime(df["date_week"])
df = df.sort_values("date_week").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ==========================================
# 2. CHRONOLOGICAL SPLIT (80% TRAIN / 20% TEST)
# ==========================================
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(f"\nTraining observations: {len(train_df)}")
print(f"Testing observations : {len(test_df)}")

# ==========================================
# 3. FIT MARKETING MIX MODEL WITH SATURATION
# ==========================================
model = MarketingMixModel(decay_rate=0.4, sat_type="log")
model.fit(train_df)

eval_metrics = model.evaluate(test_df)

print("\n========== MARKETING MIX MODEL EVALUATION (HOLD-OUT TEST) ==========")
print(f"MAE :  {eval_metrics['MAE']:.2f}")
print(f"RMSE:  {eval_metrics['RMSE']:.2f}")
print(f"R²  :  {eval_metrics['R2']:.4f}")

params = model.get_channel_parameters()
print("\n========== LEARNED CHANNEL RESPONSE PARAMETERS ==========")
print(f"Channel 1 Effectiveness (Beta 1): {params['beta_1']:.4f}")
print(f"Channel 2 Effectiveness (Beta 2): {params['beta_2']:.4f}")
print(f"Channel 1 Saturation Scale (K1): {params['scale_1']:.4f}")
print(f"Channel 2 Saturation Scale (K2): {params['scale_2']:.4f}")
print(f"Saturation Curve: {params['sat_type'].upper()}")
print(f"Adstock Decay Factor (Decay Rate): {params['decay_rate']}")
print(f"Baseline Outcome (Intercept + Controls): {params['baseline_outcome']:.2f}")

# ==========================================
# 4. PLOT ACTUAL VS PREDICTED OUTCOMES
# ==========================================
test_predictions = model.predict(test_df)
results = pd.DataFrame({
    "date_week": test_df["date_week"].values,
    "actual": test_df["y"].values,
    "predicted": test_predictions,
})

plt.figure(figsize=(12, 5))
plt.plot(results["date_week"], results["actual"], marker="o", label="Actual Outcome")
plt.plot(results["date_week"], results["predicted"], marker="x", label="Predicted Outcome (MMM)")
plt.title("Marketing Mix Model: Actual vs Predicted (Held-Out Test Set)")
plt.xlabel("Week")
plt.ylabel("Outcome (y)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot to data/
plot_path = PROJECT_ROOT / "data" / "mmm_actual_vs_predicted.png"
plt.savefig(plot_path)
plt.close()

# Save predictions
pred_path = PROJECT_ROOT / "data" / "mmm_predictions.csv"
results.to_csv(pred_path, index=False)
print(f"\nPredictions saved to {pred_path}")
print(f"Forecast plot saved to {plot_path}")