import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
url = "data.csv"
df = pd.read_csv(url)

# Selecting columns for MLP
X = df.drop('diagnosis', axis=1)  # Features
y = df['diagnosis']  # Target

X = X.fillna(np.mean(X))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adding noise to training data
X_train+=np.random.normal(0, 0.1, X_train.shape)  # Mean 0, Standard deviation 0.1


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Train the classifier on noisy data
mlp_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = mlp_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Multi-Layer Perceptron (MLP) Accuracy with Noisy Data:", accuracy)
