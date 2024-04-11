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

# Initialize MLP classifiers with different optimizers
mlp_sgd = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.0001, solver='sgd',
                        batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                        max_iter=500, tol=1e-4, early_stopping=False, random_state=42)

mlp_adam = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.0001, solver='adam',
                         batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                         max_iter=500, tol=1e-4, early_stopping=False, random_state=42)

mlp_mini_batch = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.0001, solver='adam',
                               batch_size=64, learning_rate='constant', learning_rate_init=0.001,
                               max_iter=500, tol=1e-4, early_stopping=False, random_state=42)

mlp_lbfgs = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.0001, solver='lbfgs',
                          batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                          max_iter=500, tol=1e-4, early_stopping=False, random_state=42)

# Fit the MLP classifiers
mlp_sgd.fit(X_train, y_train)
mlp_adam.fit(X_train, y_train)
mlp_mini_batch.fit(X_train, y_train)
mlp_lbfgs.fit(X_train, y_train)

# Predict on the test data
y_pred_sgd = mlp_sgd.predict(X_test)
y_pred_adam = mlp_adam.predict(X_test)
y_pred_mini_batch = mlp_mini_batch.predict(X_test)
y_pred_lbfgs = mlp_lbfgs.predict(X_test)

# Calculate accuracies
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
accuracy_adam = accuracy_score(y_test, y_pred_adam)
accuracy_mini_batch = accuracy_score(y_test, y_pred_mini_batch)
accuracy_lbfgs = accuracy_score(y_test, y_pred_lbfgs)

# Print accuracies
print("MLP with SGD Optimizer Accuracy:", accuracy_sgd)
print("MLP with Adam Optimizer Accuracy:", accuracy_adam)
print("MLP with Mini-Batch Gradient Descent Accuracy:", accuracy_mini_batch)
print("MLP with LBFGS Optimizer Accuracy:", accuracy_lbfgs)
