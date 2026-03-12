import pandas as pd

# Read CSV file
df = pd.read_csv("data.csv")

# First 5 rows
print("First 5 Records:")
print(df.head())

# Last 5 rows
print("\nLast 5 Records:")
print(df.tail())