import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVR regressor
svr = SVR()

# Define the parameter grid for SVR
param_grid_svr = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Create the GridSearchCV object for SVR
grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr)

# Fit the grid search to the data for SVR
grid_search_svr.fit(X_train, y_train)

# Best parameters found by GridSearchCV for SVR
best_params_svr = grid_search_svr.best_params_
print("Best Parameters for SVR:", best_params_svr)

# Use the best SVR model found by GridSearchCV
best_svr_model = grid_search_svr.best_estimator_

# Predict on the test data using the best SVR model
y_pred_svr = best_svr_model.predict(X_test)

# Calculate RMSE for SVR
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
print("SVR GridSearchCV RMSE:", rmse_svr)

# Define the Linear Regression model
linear_reg = LinearRegression()

# Fit the Linear Regression model
linear_reg.fit(X_train, y_train)

# Predict on the test data using Linear Regression
y_pred_linear_reg = linear_reg.predict(X_test)

# Calculate RMSE for Linear Regression
rmse_linear_reg = np.sqrt(mean_squared_error(y_test, y_pred_linear_reg))
print("Linear Regression RMSE:", rmse_linear_reg)
