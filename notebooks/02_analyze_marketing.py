import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/mmm_example.csv")


df["date_week"] = pd.to_datetime(df["date_week"])


plt.figure(figsize=(12, 5))

plt.plot(
    df["date_week"],
    df["y"],
    label="Outcome",
)

plt.title("Outcome Over Time")
plt.xlabel("Date")
plt.ylabel("Outcome (y)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))

plt.plot(
    df["date_week"],
    df["x1"],
    label="Channel x1",
)

plt.plot(
    df["date_week"],
    df["x2"],
    label="Channel x2",
)

plt.title("Marketing Channel Variables Over Time")
plt.xlabel("Date")
plt.ylabel("Channel value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



columns = [
    "y",
    "x1",
    "x2",
    "event_1",
    "event_2",
    "dayofyear",
    "t",
]

correlation = df[columns].corr()

print("\n--- CORRELATION WITH OUTCOME ---")
print(correlation["y"].sort_values(ascending=False))

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()