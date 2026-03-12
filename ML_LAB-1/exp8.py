import pandas as pd

# Read CSV file
df = pd.read_csv("data.csv")

# Statistical summary
summary = df.describe()

print("Statistical Summary:")
print(summary)