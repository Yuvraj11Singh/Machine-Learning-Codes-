import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
data = load_iris()
X = data.data

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different k values
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    plt.figure()
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
    plt.title(f"K-Means Clustering (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()