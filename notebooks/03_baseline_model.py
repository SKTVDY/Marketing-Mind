import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("data/mmm_example.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)




features = [
    "x1",
    "x2",
    "event_1",
    "event_2",
    "dayofyear",
    "t"
]

X = df[features]
y = df["y"]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))



model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully!")



y_pred = model.predict(X_test)




mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


print("\n========== MODEL COEFFICIENTS ==========")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature:12s}: {coefficient:.4f}")

print(f"{'Intercept':12s}: {model.intercept_:.4f}")



results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results["Error"] = results["Actual"] - results["Predicted"]

print("\n========== SAMPLE PREDICTIONS ==========")

print(results.head(10).round(2))


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    range(len(results)),
    results["Actual"].values,
    marker="o",
    label="Actual"
)

plt.plot(
    range(len(results)),
    results["Predicted"].values,
    marker="x",
    label="Predicted"
)

plt.title("Actual vs Predicted Outcomes")
plt.xlabel("Test Sample")
plt.ylabel("Outcome (y)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()