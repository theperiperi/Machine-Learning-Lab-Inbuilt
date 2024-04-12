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

# Create noisy data (Gaussian noise)
X_noisy = X + np.random.normal(0, 0.05, X.shape)

# Combine original data and noisy data
X_combined = np.vstack((X, X_noisy))
y_combined = np.hstack((y, y))  # Use original labels for both original and noisy data

# Create noisy data (Gaussian noise)
X_noisy = X_train + np.random.normal(0, 0.05, X_train.shape)

# Combine original data and noisy data
X_augmented = np.vstack((X_train, X_noisy))
y_augmented = np.hstack((y_train, y_train))  # Use original labels for both original and noisy data

# Create an MLP Classifier for augmented data
mlp_augmented = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.0001, solver='adam',
                              batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                              max_iter=500, tol=1e-4, early_stopping=False, random_state=42)

# Fit the model
mlp_augmented.fit(X_augmented, y_augmented)

# Predict on the test data
y_pred_augmented = mlp_augmented.predict(X_test)

# Calculate accuracy
accuracy_augmented = accuracy_score(y_test, y_pred_augmented)
print("MLP with Data Augmentation Only Accuracy:", accuracy_augmented)
