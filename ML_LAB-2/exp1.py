import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# Check missing values
print("Missing Values:")
print(df.isnull())

# Check non-missing values
print("\nNon-Missing Values:")
print(df.notnull())

# Total missing values per column
print("\nTotal Missing Values in Each Column:")
print(df.isnull().sum())