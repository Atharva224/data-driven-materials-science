
# -*- coding: utf-8 -*-
"""
@author: Atharva Sinnarkar
"""
# this is just a module to allow you to write out your results and read them 
# again
from json import dump,load
import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.linear_model import lars_path 
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":

# Task 3.1: Least Angle Regression

    # Load data
    features_df = pd.read_csv("features-bulk.csv")
    target_df = pd.read_csv("target-bulk.csv")
    
    # Convert dataframes to numpy arrays
    X = features_df.values
    y = target_df.values.ravel()  
    
    # Apply LARS with LASSO regularization
    _, _, coefs = lars_path(X, y, method='lasso', verbose=True)
    
    # Get the most important features
    important_features_indices = np.argsort(np.abs(coefs.sum(axis=1)))[-10:]
    important_features = features_df.columns[important_features_indices]
    
    print("Top 10 most important features from lasso with lars:")
    print("")
    print(", ".join(important_features))
    print("")
    
    # Perform polynomial + interaction expansion
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Apply LARS with LASSO regularization on expanded features
    _, _, coefs_poly = lars_path(X_poly, y, method='lasso', verbose=True)
    
    # Get the most important features from expanded features
    important_features_indices_poly = np.argsort(np.abs(coefs_poly.sum(axis=1)))[-10:]
    important_features_poly = poly.get_feature_names_out(features_df.columns)[important_features_indices_poly]
    print("")
    print("\nTop 10 most important features after polynomial + interaction expansion:")
    print("")
    print(", ".join(important_features_poly.tolist()))
    print("")

    ####################################################################################################

   # Task 3.2: Recursive Feature Elimination

    # Define models
    tree_model = DecisionTreeRegressor()
    linear_model = Lasso(max_iter=10000)  # Set max_iter to avoid convergence warning

    # RFE with tree model
    rfe_tree = RFE(tree_model, n_features_to_select=10)
    rfe_tree.fit(X, y)
    important_features_tree = features_df.columns[rfe_tree.support_]
    
    print("Top 10 most important features using Recursive Feature Elimination with a tree model:")
    print("")
    print(", ".join(important_features_tree))
    print("")
    
    # Standardize features for linear model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # RFE with linear model (LASSO)
    rfe_linear = RFE(linear_model, n_features_to_select=10)
    rfe_linear.fit(X_scaled, y)
    important_features_linear = features_df.columns[rfe_linear.support_]
    
    print("Top 10 most important features using Recursive Feature Elimination with a linear model (LASSO):")
    print("")
    print(", ".join(important_features_linear))

####################################################################################################
# Plot coefficient paths for LASSO regularization using LARS
plt.figure(figsize=(10, 6))
for coef in coefs.T:
    plt.plot(np.abs(coef))
plt.xlabel('Iteration')
plt.ylabel('Coefficient Magnitude')
plt.title('Coefficient Paths for LASSO Regularization (LARS)')
plt.show()



# Plot histograms for the distribution of top important features
plt.figure(figsize=(12, 8))
for i, feature in enumerate(important_features):
    plt.subplot(2, 5, i+1)
    plt.hist(features_df[feature], bins=20)
    plt.title(feature, fontsize=7)  # Adjust the font size as needed
plt.tight_layout()
plt.show()


# Plot model performance vs. number of features
num_features = range(1, len(important_features) + 1)
mse_results = []
r_squared_results = []

for num in num_features:
    # Train models with different number of features
    # Example: linear regression with top num features
    model = LinearRegression()
    rfe_linear = RFE(model, n_features_to_select=num)
    rfe_linear.fit(X_scaled, y)
    X_selected = rfe_linear.transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate MSE and R-squared
    mse = np.mean((y_pred - y_test) ** 2)
    r_squared = model.score(X_test, y_test)
    
    mse_results.append(mse)
    r_squared_results.append(r_squared)

plt.figure(figsize=(10, 6))
plt.plot(num_features, mse_results, label='MSE')
plt.plot(num_features, r_squared_results, label='R-squared')
plt.xlabel('Number of Features')
plt.ylabel('Performance Metric')
plt.title('Model Performance vs. Number of Features')
plt.legend()
plt.show()

# Create a heatmap of correlation matrix for selected features
selected_features_df = features_df[important_features]
correlation_matrix = selected_features_df.corr()

plt.figure(figsize=(10, 8))
plt.title('Correlation Heatmap of Selected Features')
heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(ticks=range(len(important_features)), labels=important_features, rotation=45, ha='right')
plt.yticks(ticks=range(len(important_features)), labels=important_features, rotation=0)
plt.tight_layout()  # Adjust layout to ensure everything fits
plt.show()




pass
    