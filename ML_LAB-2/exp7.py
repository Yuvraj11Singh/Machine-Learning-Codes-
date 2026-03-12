import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# Line Plot
plt.plot(df["Year"], df["Sales"])
plt.title("Sales Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.show()

# Bar Plot
df["Category"].value_counts().plot(kind="bar")
plt.title("Category Comparison")
plt.show()