import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")

num_cols = ['math score', 'reading score', 'writing score']

# Correlation
correlation = df[num_cols].corr()
print("Correlation Matrix:")
print(correlation)

# Covariance
covariance = df[num_cols].cov()
print("\nCovariance Matrix:")
print(covariance)