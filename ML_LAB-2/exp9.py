import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

plt.hist(df["Salary"], bins=10)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()