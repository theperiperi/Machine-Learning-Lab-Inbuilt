import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the breast cancer dataset
df = pd.read_csv("data.csv")

# Selecting columns for clustering
X_cluster = df[['radius_mean', 'texture_mean']]

# Perform KMeans clustering with 2 clusters (as an example)
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X_cluster)

# Calculate KMeans centroids
centroids = kmeans.cluster_centers_
print("Centroids:", centroids)

# Visualize the clusters
plt.scatter(X_cluster['radius_mean'], X_cluster['texture_mean'], c=df['cluster'], cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.title('K-Means Clustering')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()
