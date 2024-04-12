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

# Create an MLP Classifier with L1 Regularization
mlp_l1 = MLPClassifier(hidden_layer_sizes=(100, 50),alpha=0.01,max_iter=500 ,random_state=42)

# Fit the model
mlp_l1.fit(X_train, y_train)

# Predict on the test data
y_pred_l1 = mlp_l1.predict(X_test)

# Calculate accuracy
accuracy_l1 = accuracy_score(y_test, y_pred_l1)
print("MLP with L1 Regularization Accuracy:", accuracy_l1)
