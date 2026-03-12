import pandas as pd

df = pd.read_csv("data.csv")

# Replace numerical missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Replace categorical missing values with constant
df.fillna("Unknown", inplace=True)

print(df)