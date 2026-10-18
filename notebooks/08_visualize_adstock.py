import pandas as pd
import matplotlib.pyplot as plt

# Load the saved validation results
results = pd.read_csv("data/adstock_validation_results.csv")

# Sort by decay rate for a clearer graph
results = results.sort_values("decay_rate")

# Display results
print(results.to_string(index=False))

# Plot MAE against decay rate
plt.figure(figsize=(9, 5))

plt.plot(
    results["decay_rate"],
    results["MAE"],
    marker="o"
)

plt.xlabel("Adstock Decay Rate")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Effect of Adstock Decay Rate on Prediction Error")
plt.grid(True)

plt.tight_layout()
plt.savefig("data/adstock_decay_comparison.png", dpi=300)
plt.show()

print("\nGraph saved to data/adstock_decay_comparison.png")