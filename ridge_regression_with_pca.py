# ridge_regression_with_pca.py

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pca import PrincipalComponentAnalysis

# Load data
features_df = pd.read_csv("features-bulk.csv")
target_df = pd.read_csv("target-bulk.csv")
X = features_df.values
y = target_df.values.ravel()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
n_components = 10  # Number of principal components to use
pca = PrincipalComponentAnalysis(n_components=n_components)
pca.train(X_scaled)
X_pca = pca.transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=42)

# Linear Ridge Regression
ridge_params = {'alpha': [0.01, 0.1, 1, 2, 10]}
ridge = Ridge(max_iter=38000)
ridge_grid = GridSearchCV(ridge, param_grid=ridge_params, cv=5)
ridge_grid.fit(X_train, y_train)
ridge_best = ridge_grid.best_estimator_

# Kernel Ridge Regression
kernel_ridge_params = {
    'alpha': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.01, 0.1, 1, 10],
    'degree': [2, 3, 4]
}
kernel_ridge = KernelRidge()
kernel_ridge_grid = GridSearchCV(kernel_ridge, param_grid=kernel_ridge_params, cv=5)
kernel_ridge_grid.fit(X_train, y_train)
kernel_ridge_best = kernel_ridge_grid.best_estimator_

# Evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

# Evaluate Linear Ridge Regression
ridge_mse, ridge_r2 = evaluate_model(ridge_best, X_train, X_test, y_train, y_test)

# Evaluate Kernel Ridge Regression
kernel_ridge_mse, kernel_ridge_r2 = evaluate_model(kernel_ridge_best, X_train, X_test, y_train, y_test)

# Print results
print(f"Linear Ridge Regression: MSE = {ridge_mse}, R2 = {ridge_r2}")
print(f"Kernel Ridge Regression: MSE = {kernel_ridge_mse}, R2 = {kernel_ridge_r2}")

# Plot results
plt.figure(figsize=(12, 6))

# Linear Ridge Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, ridge_best.predict(X_test), color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Ridge Regression')

# Kernel Ridge Regression
plt.subplot(1, 2, 2)
plt.scatter(y_test, kernel_ridge_best.predict(X_test), color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Kernel Ridge Regression')

plt.tight_layout()
plt.show()
