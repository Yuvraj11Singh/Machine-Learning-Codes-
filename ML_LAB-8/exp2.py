import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
data = load_iris()
X = data.data

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dendrogram
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.show()

# Clustering for k values
for k in [2, 3, 4]:
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_scaled)
    
    plt.figure()
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
    plt.title(f"Agglomerative Clustering (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()