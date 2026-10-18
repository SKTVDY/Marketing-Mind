import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




df = pd.read_csv("data/mmm_example.csv")

df["date_week"] = pd.to_datetime(df["date_week"])
df = df.sort_values("date_week").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)



def calculate_adstock(series, decay_rate=0.5):
    """
    Previous marketing activity carries over
    into subsequent weeks.

    decay_rate controls how quickly the
    previous activity loses its influence.
    """
    values = series.to_numpy()
    adstock = np.zeros(len(values))

    adstock[0] = values[0]

    for i in range(1, len(values)):
        adstock[i] = values[i] + decay_rate * adstock[i - 1]

    return adstock


df["x1_adstock"] = calculate_adstock(df["x1"], 0.5)
df["x2_adstock"] = calculate_adstock(df["x2"], 0.5)


def saturation(series):
    return series / (series.max() + 1e-8)

df["x1_saturated"] = saturation(df["x1_adstock"])
df["x2_saturated"] = saturation(df["x2_adstock"])



features = [
    "x1_saturated",
    "x2_saturated",
    "event_1",
    "event_2",
    "dayofyear",
    "t"
]

X = df[features]
y = df["y"]



split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))




model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\n========== MARKETING MIX MODEL ==========")

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")



print("\n========== MODEL COEFFICIENTS ==========")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature:20s}: {coefficient:.4f}")

print(f"{'Intercept':20s}: {model.intercept_:.4f}")




results = pd.DataFrame({
    "date_week": df["date_week"].iloc[split_index:].values,
    "actual": y_test.values,
    "predicted": y_pred
})

plt.figure(figsize=(12, 5))

plt.plot(
    results["date_week"],
    results["actual"],
    marker="o",
    label="Actual"
)

plt.plot(
    results["date_week"],
    results["predicted"],
    marker="x",
    label="Predicted"
)

plt.title("Marketing Mix Model: Actual vs Predicted")
plt.xlabel("Week")
plt.ylabel("Outcome (y)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()




results.to_csv(
    "data/mmm_predictions.csv",
    index=False
)

print("\nPredictions saved to data/mmm_predictions.csv")