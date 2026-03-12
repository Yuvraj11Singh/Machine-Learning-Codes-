import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# Dataset info
print(df.info())

# Rows and columns
print("Shape:", df.shape)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())