import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Display first few records
print("First 5 Records:")
print(df.head())

# Identify numerical and categorical columns
print("\nNumerical Columns:")
print(df.select_dtypes(include=np.number).columns)

print("\nCategorical Columns:")
print(df.select_dtypes(include='object').columns)

# Select numerical features
num_cols = ['math score', 'reading score', 'writing score']

# Measures of Central Tendency
print("\nMean:")
print(df[num_cols].mean())

print("\nMedian:")
print(df[num_cols].median())

print("\nMode:")
print(df[num_cols].mode().iloc[0])

# Measures of Dispersion
print("\nMinimum:")
print(df[num_cols].min())

print("\nMaximum:")
print(df[num_cols].max())

print("\nSum:")
print(df[num_cols].sum())

print("\nVariance:")
print(df[num_cols].var())

print("\nStandard Deviation:")
print(df[num_cols].std())