import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("co2_emission.csv")

print(df.head())
print(df.info())

# Correlation heatmap
corr = df[["VOLUME", "WEIGHT", "CO2"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots for outliers
df[["VOLUME", "WEIGHT", "CO2"]].boxplot()
plt.title("Boxplot for Outliers")
plt.show()

# Visualize relationships
plt.scatter(df["VOLUME"], df["CO2"])
plt.xlabel("Volume")
plt.ylabel("CO2")
plt.show()

plt.scatter(df["WEIGHT"], df["CO2"])
plt.xlabel("Weight")
plt.ylabel("CO2")
plt.show()

# Train-test split
X = df[["VOLUME", "WEIGHT"]]
y = df["CO2"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Prediction
y_pred = model.predict(X_test)

# True vs predicted
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted CO2")
plt.show()

# Error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)