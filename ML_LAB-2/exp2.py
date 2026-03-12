import pandas as pd

df = pd.read_csv("data.csv")

print("Original Dataset Shape:", df.shape)

# Remove rows with missing values
clean_df = df.dropna()

print("Dataset Shape After Removing Missing Values:", clean_df.shape)