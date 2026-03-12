import pandas as pd

df = pd.read_csv("data.csv")

# Rename columns
df = df.rename(columns={
    "Name": "Student_Name",
    "Marks": "Student_Marks"
})

print("Updated DataFrame:")
print(df.head())