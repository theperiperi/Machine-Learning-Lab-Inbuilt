import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the breast cancer dataset
url = "data.csv"
df = pd.read_csv(url)

# Selecting columns for Perceptron
X = df.drop('diagnosis', axis=1)  # Features
y = df['diagnosis']  # Target

X = X.fillna(np.mean(X))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Perceptron classifier
perceptron = Perceptron(random_state=42)

# Train the classifier
perceptron.fit(X_train, y_train)

# Predict on the test data
y_pred = perceptron.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Perceptron Accuracy:", accuracy)

# Apply PCA to visualize in 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting the decision boundary
plt.figure(figsize=(8, 6))
colors = ['blue', 'red']
lw = 2

for color, target_name in zip(colors, ['B', 'M']):
    plt.scatter(X_pca[y == target_name, 0], X_pca[y == target_name, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot decision boundary
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.colorbar()
plt.show()
