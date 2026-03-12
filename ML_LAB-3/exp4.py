import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("StudentsPerformance.csv")

# Histogram
df[['math score', 'reading score', 'writing score']].hist(bins=10)
plt.suptitle("Histogram of Student Scores")
plt.show()

# Boxplot
df[['math score', 'reading score', 'writing score']].boxplot()
plt.title("Boxplot of Student Scores")
plt.show()