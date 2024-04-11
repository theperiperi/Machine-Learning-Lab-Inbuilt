import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
url = "data.csv"
df = pd.read_csv(url)

# Selecting columns for Perceptron
X = df.drop('diagnosis', axis=1)  # Features
y = df['diagnosis']  # Target

X=X.fillna(np.mean(X))
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
