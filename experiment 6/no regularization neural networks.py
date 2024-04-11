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

# Create an MLP Classifier without Regularization
mlp_no_reg = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.0, solver='adam',
                           batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                           max_iter=500, tol=1e-4, early_stopping=False, random_state=42)

# Fit the model
mlp_no_reg.fit(X_train, y_train)

# Predict on the test data
y_pred_no_reg = mlp_no_reg.predict(X_test)

# Calculate accuracy
accuracy_no_reg = accuracy_score(y_test, y_pred_no_reg)
print("MLP without Regularization Accuracy:", accuracy_no_reg)
