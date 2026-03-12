import pandas as pd

df = pd.read_csv("data.csv")

# Replace incorrect values
df["Gender"] = df["Gender"].replace({
    "M": "Male",
    "F": "Female"
})

print("Corrected Data:")
print(df.head())