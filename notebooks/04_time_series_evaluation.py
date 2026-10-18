import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("data/mmm_example.csv")

df["date_week"] = pd.to_datetime(df["date_week"])


features = [
    "x1",
    "x2",
    "event_1",
    "event_2",
    "dayofyear",
    "t",
]

X = df[features]
y = df["y"]


split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n--- CHRONOLOGICAL EVALUATION ---")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")


results = pd.DataFrame({
    "date_week": df["date_week"].iloc[split_index:].values,
    "actual": y_test.values,
    "predicted": y_pred,
})

# 8. Plot results
plt.figure(figsize=(12, 5))

plt.plot(
    results["date_week"],
    results["actual"],
    marker="o",
    label="Actual",
)

plt.plot(
    results["date_week"],
    results["predicted"],
    marker="x",
    label="Predicted",
)

plt.title("Actual vs Predicted Outcomes — Future Weeks")
plt.xlabel("Week")
plt.ylabel("Outcome (y)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()