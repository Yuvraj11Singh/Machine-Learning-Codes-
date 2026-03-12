import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

plt.scatter(df["Age"], df["Salary"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary Relationship")
plt.show()