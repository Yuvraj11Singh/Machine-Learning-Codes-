import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")

num_cols = ['math score', 'reading score', 'writing score']

# Quartiles
Q1 = df[num_cols].quantile(0.25)
Q2 = df[num_cols].quantile(0.50)
Q3 = df[num_cols].quantile(0.75)

print("First Quartile (Q1):")
print(Q1)

print("\nSecond Quartile (Median / Q2):")
print(Q2)

print("\nThird Quartile (Q3):")
print(Q3)