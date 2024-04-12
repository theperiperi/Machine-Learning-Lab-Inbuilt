import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv("data.csv")

# Load breast cancer dataset
X=df.drop("diagnosis",axis=1)
y=df["diagnosis"]

X=X.fillna(np.mean(X))
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

import numpy as np

# Create noisy data (Gaussian noise)
X_noisy = X + np.random.normal(0, 0.05, X.shape)

# Combine original data and noisy data
X_combined = np.vstack((X, X_noisy))
y_combined = np.hstack((y, y))  # Use original labels for both original and noisy data

# Split the combined data into train and test sets
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
# Create an MLP Classifier with Early Stopping
mlp_early_stop = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,early_stopping=True)

# Fit the model
mlp_early_stop.fit(X_train, y_train)

# Predict on the test data
y_pred_early_stop = mlp_early_stop.predict(X_test)

# Calculate accuracy
accuracy_early_stop = accuracy_score(y_test, y_pred_early_stop)
print("MLP with Early Stopping Accuracy:", accuracy_early_stop)
