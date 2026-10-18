import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/mmm_example.csv")

df["date_week"] = pd.to_datetime(df["date_week"])
df = df.sort_values("date_week").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. ADSTOCK FUNCTION
# ==========================================

def calculate_adstock(series, decay_rate):
    values = series.to_numpy()
    adstock = np.zeros(len(values))

    adstock[0] = values[0]

    for i in range(1, len(values)):
        adstock[i] = values[i] + decay_rate * adstock[i - 1]

    return adstock


# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

y_train = train["y"]
y_test = test["y"]

print("\nTraining observations:", len(train))
print("Testing observations:", len(test))


# ==========================================
# 4. TEST DIFFERENT DECAY RATES
# ==========================================

decay_rates = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]

results = []

for decay in decay_rates:

    # Calculate adstock on the complete chronological
    # series so earlier activity carries into later weeks.
    x1_adstock = calculate_adstock(df["x1"], decay)
    x2_adstock = calculate_adstock(df["x2"], decay)

    # Split the transformed data chronologically.
    X = pd.DataFrame({
        "x1_saturated": x1_adstock / (x1_adstock.max() + 1e-8),
        "x2_saturated": x2_adstock / (x2_adstock.max() + 1e-8),
        "event_1": df["event_1"],
        "event_2": df["event_2"],
        "dayofyear": df["dayofyear"],
        "t": df["t"]
    })

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "decay_rate": decay,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\n========== ADSTOCK TUNING RESULTS ==========")
print(results_df.to_string(index=False))

best_decay = results_df.iloc[0]["decay_rate"]

print("\nBest decay rate by MAE:", best_decay)

results_df.to_csv("data/adstock_tuning_results.csv", index=False)

print("\nResults saved to data/adstock_tuning_results.csv")