import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/mmm_example.csv")


print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- DATASET SHAPE ---")
print(df.shape)


print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())


print("\n--- DATA TYPES ---")
print(df.dtypes)


print("\n--- MISSING VALUES ---")
print(df.isnull().sum())


print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())