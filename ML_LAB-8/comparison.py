import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Load dataset
data = load_iris()
X = data.data

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3

# Models
kmeans = KMeans(n_clusters=k, random_state=42)
agg = AgglomerativeClustering(n_clusters=k)

k_labels = kmeans.fit_predict(X_scaled)
a_labels = agg.fit_predict(X_scaled)

# Side-by-side plots
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=k_labels)
plt.title("K-Means")

plt.subplot(1,2,2)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=a_labels)
plt.title("Agglomerative")

plt.show()