import pandas as pd

# Dictionary
data = {
    "Name": ["Aman", "Riya", "Rahul", "Neha"],
    "Marks": [85, 90, 78, 92]
}

# Create DataFrame
df = pd.DataFrame(data)

print("DataFrame:")
print(df)

print("\nData Types:")
print(df.dtypes)