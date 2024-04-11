import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the breast cancer dataset
url = "data.csv"
df = pd.read_csv(url)

# Selecting columns for PCA
X = df[['radius_mean', 'texture_mean']]

# Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
total_variance_explained = sum(explained_variance_ratio)

print("Explained Variance Ratio:", explained_variance_ratio)
print("Total Variance Explained by 2 Components:", total_variance_explained)

# Visualize PCA components
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8, edgecolors='w')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()
