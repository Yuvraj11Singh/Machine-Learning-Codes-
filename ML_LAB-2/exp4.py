import pandas as pd

df = pd.read_csv("data.csv")

print("Original Data Types:")
print(df.dtypes)

# Example conversion
df["Age"] = df["Age"].astype(int)

print("\nUpdated Data Types:")
print(df.dtypes)