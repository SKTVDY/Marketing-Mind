# Marketing Mind — "Predict, Then Optimize" MMM Platform

Marketing Mind is an end-to-end Marketing Mix Modeling (MMM) and budget allocation engine built in Python. It models historical carryover effects (**Adstock**) and non-linear diminishing returns (**Saturation Curves**), tunes hyperparameters using leakage-safe **TimeSeriesSplit cross-validation**, and feeds learned empirical regression parameters directly into **SciPy SLSQP** to solve optimal cross-channel spend distributions.

---

## 🚀 Key Upgrades & Highlights

1. **"Predict, Then Optimize" Architecture:**
   - Replaced hardcoded response parameters with empirical coefficients ($\beta_1, \beta_2$) and saturation parameters ($K_1, K_2$) learned directly from the regression model.
   - Dynamically optimizes non-linear concave channel return functions subject to total budget and channel minimum/maximum constraints.

2. **Diminishing Returns Saturation Curves:**
   - Implemented non-linear saturation transformations alongside geometric adstock:
     - **Logarithmic Saturation:** $\ln(1 + x / K)$
     - **Hill / Michaelis-Menten:** $x / (K + x)$
     - **Square-Root:** $\sqrt{x / K}$
   - Prevents lookahead bias by fitting saturation scaling parameters ($K$) strictly on training observations.

3. **Leakage-Safe Validation Strategy:**
   - **Chronological Holdout Split:** 80% train (143 weeks) vs. 20% test (36 weeks).
   - **TimeSeriesSplit CV:** Hyperparameters (decay rate $\lambda$ and saturation curves) are tuned using 5-fold expanding-window cross-validation on the training set only.
   - **Unbiased Assessment:** The held-out test split is evaluated **strictly once** to report honest generalization performance ($R^2$, MAE, RMSE).

4. **Interactive Streamlit Dashboard:**
   - Real-time SLSQP budget re-allocation with side-by-side benchmark comparison (Equal Split vs. Optimized).
   - Visual diminishing returns response curves with optimal spend markers.
   - Out-of-sample forecast comparisons and scenario simulator.

---

## 📊 Performance Benchmarks (Held-Out Test Set)

| Model Configuration | Decay ($\lambda$) | Saturation Curve | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Best Tuned Model (Adstock)** | **0.5** | **None** | **492.93** | **687.73** | **0.4823** |
| **Adstock + Log Saturation** | **0.4** | **Log** | **507.61** | **695.37** | **0.4708** |
| Baseline (No Adstock, No Saturation) | 0.0 | None | 574.83 | 750.50 | 0.3835 |

> **Impact:** +25.8% lift in test $R^2$ and 14.2% reduction in prediction error (MAE) over the standard linear baseline.

---

## 🛠️ Tech Stack

- **Core ML & Math:** Python, Scikit-learn, SciPy (`minimize` / SLSQP), NumPy, Pandas
- **Validation:** `TimeSeriesSplit` cross-validation (5 folds)
- **Data Visualization & UI:** Streamlit, Matplotlib
- **Testing:** Pytest

---

## 📁 Project Structure

```text
Marketing Mind/
├── app/
│   └── app.py                      # Interactive Streamlit dashboard
├── data/
│   ├── mmm_example.csv             # Weekly marketing spend & outcome dataset (179 weeks)
│   ├── adstock_tuning_results.csv   # TimeSeriesSplit CV grid-search scores
│   ├── adstock_validation_results.csv # Final unbiased test set evaluation metrics
│   └── mmm_predictions.csv         # Out-of-sample predictions
├── notebooks/
│   ├── 01_explore_data.py
│   ├── 02_analyze_marketing.py
│   ├── 03_baseline_model.py
│   ├── 04_time_series_evaluation.py
│   ├── 05_marketing_mix_model.py   # Full MMM training pipeline & parameter extraction
│   ├── 06_tune_adstock.py          # TimeSeriesSplit CV tuning on train split
│   ├── 07_validate_adstock.py      # Final one-time test set evaluation
│   └── 08_visualize_adstock.py
├── src/
│   ├── mmm_transforms.py           # Adstock carryover & saturation curves (Log, Hill, Sqrt)
│   ├── model_trainer.py            # MarketingMixModel class & TimeSeriesSplit CV engine
│   └── budget_optimizer.py         # SLSQP constrained optimizer connected to fitted model
├── tests/
│   ├── test_budget_optimizer.py    # Unit tests for budget allocation & constraints
│   └── test_model_pipeline.py      # Tests for adstock, saturation, & model fitting
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

1. **Activate virtual environment:**
   ```bash
   .\.venv\Scripts\activate
   ```
2. **Run TimeSeriesSplit Hyperparameter Tuning:**
   ```bash
   python notebooks/06_tune_adstock.py
   ```
3. **Run Final Unbiased Test Set Evaluation:**
   ```bash
   python notebooks/07_validate_adstock.py
   ```
4. **Launch Streamlit Dashboard:**
   ```bash
   streamlit run app/app.py
   ```
5. **Run Test Suite:**
   ```bash
   pytest
   ```
