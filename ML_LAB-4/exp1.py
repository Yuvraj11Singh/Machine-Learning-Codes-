import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("TvMarketing.csv")

# Display dataset structure
print(df.head())
print(df.info())

# Visualize data
plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Budget")
plt.ylabel("Sales")
plt.title("TV Budget vs Sales")
plt.show()

# Train-test split
X = df[["TV"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Best fit line
plt.scatter(X, y)
plt.plot(X, model.predict(X), color="red")
plt.xlabel("TV Budget")
plt.ylabel("Sales")
plt.title("Regression Line")
plt.show()

# Prediction
y_pred = model.predict(X_test)

# Actual vs predicted
print(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).head())

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)