
# -*- coding: utf-8 -*-
"""

@author: Atharva Sinnarkar
"""

# this is just a module to allow you to write out your results and read them 
# again
from json import dump,load
import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV,train_test_split 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,\
                             RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
import seaborn as sns

####################################################################################################

#Task 2.2

# Load data
features_df = pd.read_csv("features-bulk.csv")
target_df = pd.read_csv("target-bulk.csv")

# Extract features and target
X = features_df.values
y = target_df.values.ravel()  
print("size of feature dataset is:", X.shape)
print("size of target dataset is:", y.shape)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("size of scaled feature dataset is:",X_scaled.shape)

# Fit Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_scaled, y)

# Define parameter grid for grid search
lasso_params = {'alpha': [0.01, 0.02, 0.1, 1, 10]}
ridge_params = {'alpha': [0.01, 0.1, 1, 2, 10]}

# Perform grid search for Lasso
lasso = Lasso(max_iter=38000)
lasso_grid = GridSearchCV(lasso, param_grid=lasso_params, cv=5)
lasso_grid.fit(X_scaled, y)
df_lasso = pd.DataFrame(lasso_grid.cv_results_)
print("Cross Validation Table For Lasso:\n",df_lasso)
lasso_best_alpha = lasso_grid.best_params_['alpha']
lasso_best = lasso_grid.best_estimator_

# Perform grid search for Ridge
ridge = Ridge(max_iter=38000)
ridge_grid = GridSearchCV(ridge, param_grid=ridge_params, cv=5)
ridge_grid.fit(X_scaled, y)
df_ridge = pd.DataFrame(ridge_grid.cv_results_)
print("Cross Validation Table For Ridge:\n",df_ridge)
ridge_best_alpha = ridge_grid.best_params_['alpha']
ridge_best = ridge_grid.best_estimator_

# Extract and save the ten most important features (largest norm of weights) for Linear Regression
linear_selector = SelectFromModel(linear_model, max_features=10)
linear_selector.fit(X_scaled, y)
linear_selected_features = features_df.columns[linear_selector.get_support()].tolist()

# Extract and save the ten most important features (largest norm of weights) for Lasso
lasso_selector = SelectFromModel(lasso_best, max_features=10)
lasso_selector.fit(X_scaled, y)
lasso_selected_features = features_df.columns[lasso_selector.get_support()].tolist()

# Extract and save the ten most important features (largest norm of weights) for Ridge
ridge_selector = SelectFromModel(ridge_best, max_features=10)
ridge_selector.fit(X_scaled, y)
ridge_selected_features = features_df.columns[ridge_selector.get_support()].tolist()

# Print the selected features and optimal alpha values
# Print the top 10 features selected by Linear Regression
print("Top 10 features selected by Linear Regression:", linear_selected_features)
print("")
print("Top 10 features selected by Lasso:", lasso_selected_features)
print("")
print("Top 10 features selected by Ridge:", ridge_selected_features)
print("")
print("Optimal alpha for Lasso:", lasso_best_alpha)
print("")
print("Optimal alpha for Ridge:", ridge_best_alpha)
print("")

####################################################################################################

#Graph plotting 

# Predictions
linear_predictions = linear_model.predict(X_scaled)
lasso_predictions = lasso_best.predict(X_scaled)
ridge_predictions = ridge_best.predict(X_scaled)

# Plotting
plt.figure(figsize=(12, 6))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y, linear_predictions, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Features')
plt.ylabel('Target')
plt.title('Linear Regression')

# Lasso Regression
plt.subplot(1, 3, 2)
plt.scatter(y, lasso_predictions, color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Features')
plt.ylabel('Target')
plt.title('Lasso Regression')

# Ridge Regression
plt.subplot(1, 3, 3)
plt.scatter(y, ridge_predictions, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Features')
plt.ylabel('Target')
plt.title('Ridge Regression')

plt.tight_layout()
plt.show()


####################################################################################################
#cross validation ploting for lasso and ridge

plt.figure(figsize=(12, 6))

# Lasso Cross-Validation Results
plt.subplot(1, 2, 1)
plt.plot(lasso_params['alpha'], lasso_grid.cv_results_['mean_test_score'], marker='o')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')
plt.title('Lasso Cross-Validation Scores')
plt.xscale('log')

# Ridge Cross-Validation Results
plt.subplot(1, 2, 2)
plt.plot(ridge_params['alpha'], ridge_grid.cv_results_['mean_test_score'], marker='o')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')
plt.title('Ridge Cross-Validation Scores')
plt.xscale('log')

plt.tight_layout()
plt.show()



####################################################################################################


# Task 2.3: Polynomial Expansion

# Define the order of polynomial expansion (e.g., 2 for quadratic features)
poly_order = 2

# Perform polynomial feature expansion
poly = PolynomialFeatures(degree=poly_order, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
print("Size of polynomial feature dataset is:", X_poly.shape)
print("")

# Fit Linear Regression model with polynomial features
linear_model_poly = LinearRegression()
linear_model_poly.fit(X_poly, y)

# Extract and save the ten most important features/terms for Linear Regression
linear_coef_abs = np.abs(linear_model_poly.coef_)
top_linear_indices = np.argsort(linear_coef_abs)[-10:][::-1]
top_linear_features = [poly.get_feature_names_out(input_features=features_df.columns)[i] for i in top_linear_indices]
print("Top 10 features/terms selected by Linear Regression with polynomial expansion:", top_linear_features)
print("")

# Fit Lasso model with polynomial features
lasso_poly = Lasso(alpha=lasso_best_alpha, max_iter=38000)
lasso_poly.fit(X_poly, y)

# Extract and save the ten most important features/terms for Lasso
lasso_coef_abs = np.abs(lasso_poly.coef_)
top_lasso_indices = np.argsort(lasso_coef_abs)[-10:][::-1]
top_lasso_features = [poly.get_feature_names_out(input_features=features_df.columns)[i] for i in top_lasso_indices]
print("Top 10 features/terms selected by Lasso with polynomial expansion:", top_lasso_features)
print("")

# Fit Ridge model with polynomial features
ridge_poly = Ridge(alpha=ridge_best_alpha, max_iter=38000)
ridge_poly.fit(X_poly, y)

# Extract and save the ten most important features/terms for Ridge
ridge_coef_abs = np.abs(ridge_poly.coef_)
top_ridge_indices = np.argsort(ridge_coef_abs)[-10:][::-1]
top_ridge_features = [poly.get_feature_names_out(input_features=features_df.columns)[i] for i in top_ridge_indices]
print("Top 10 features/terms selected by Ridge with polynomial expansion:", top_ridge_features)
print("")

####################################################################################################

# Polynomial Feature Importances Plot for Linear, Lasso, and Ridge Regression

# Polynomial Feature Importances Plot for Linear Regression
plt.figure(figsize=(8, 6))
plt.barh(top_linear_features, linear_coef_abs[top_linear_indices], color='blue')
plt.xlabel('Coefficient Value')
plt.title('Linear Regression (Polynomial Features)')
plt.yticks(np.arange(len(top_linear_features)), top_linear_features)
plt.tight_layout()
plt.show()

# Polynomial Feature Importances Plot for Lasso
plt.figure(figsize=(8, 6))
plt.barh(top_lasso_features, lasso_coef_abs[top_lasso_indices], color='red')
plt.xlabel('Coefficient Value')
plt.title('Lasso Regression (Polynomial Features)')
plt.yticks(np.arange(len(top_lasso_features)), top_lasso_features)
plt.tight_layout()
plt.show()

# Polynomial Feature Importances Plot for Ridge
plt.figure(figsize=(8, 6))
plt.barh(top_ridge_features, ridge_coef_abs[top_ridge_indices], color='green')
plt.xlabel('Coefficient Value')
plt.title('Ridge Regression (Polynomial Features)')
plt.yticks(np.arange(len(top_ridge_features)), top_ridge_features)
plt.tight_layout()
plt.show()


####################################################################################################

# Task 2.4: Tree Regressors

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Define the parameter grid for grid search on decision tree depth
tree_params = {'max_depth': [1, 3, 5, 7, 10, 15]}

# Perform grid search for decision tree regressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_grid = GridSearchCV(tree_reg, param_grid=tree_params, cv=5)
tree_grid.fit(X_train, y_train)
df_tree = pd.DataFrame(tree_grid.cv_results_)
print("Cross Validation Table For Decision Tree:\n", df_tree)
tree_best_depth = tree_grid.best_params_['max_depth']
tree_best = tree_grid.best_estimator_

# Fit the selected modification method to the decision tree regressor
# Choose one modification method (Adaboost, Gradient-Boost, Hist-Boost)
base_tree = DecisionTreeRegressor(max_depth=tree_best_depth)
boosting_method = AdaBoostRegressor(random_state=42)
boosting_method.base_estimator_ = base_tree
boosting_method.fit(X_train, y_train)


# Extract and save the ten most important features for the selected modification method
boosting_features_importance = boosting_method.feature_importances_
top_boosting_indices = np.argsort(boosting_features_importance)[-10:][::-1]
top_boosting_features = features_df.columns[top_boosting_indices].tolist()
print("Top 10 features selected by the modification method:", top_boosting_features)
print("")

####################################################################################################

# Get the feature importances for the decision tree model
tree_importances = tree_best.feature_importances_
top_tree_indices = np.argsort(tree_importances)[-10:][::-1]
top_tree_features = features_df.columns[top_tree_indices]
top_tree_importances = tree_importances[top_tree_indices]

# Plot the top 10 feature importances for the decision tree model
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(top_tree_features, top_tree_importances, color='purple')
plt.xlabel('Feature Importance')
plt.title('Decision Tree Feature Importances')

# Plot the top 10 feature importances for the AdaBoost model
plt.subplot(1, 2, 2)
plt.barh(top_boosting_features, boosting_features_importance[top_boosting_indices], color='orange')
plt.xlabel('Feature Importance')
plt.title('AdaBoost Feature Importances')

plt.tight_layout()
plt.show()


####################################################################################################

# Task 2.5: Kernel Ridge Regression

# Define parameter grids for grid search
kernel_ridge_params = {'alpha': [0.01, 0.1, 1, 10],
                       'kernel': ['linear', 'rbf', 'poly'],
                       'gamma': [0.01, 0.1, 1, 10],
                       'degree': [2, 3, 4]}

# Perform grid search for Kernel Ridge Regression
kernel_ridge = KernelRidge()
kernel_ridge_grid = GridSearchCV(kernel_ridge, param_grid=kernel_ridge_params, cv=5)
kernel_ridge_grid.fit(X_scaled, y)
df_kernel_ridge = pd.DataFrame(kernel_ridge_grid.cv_results_)
print("Cross Validation Table For Kernel Ridge Regression:\n", df_kernel_ridge)

# Extract best parameters
kernel_ridge_best_params = kernel_ridge_grid.best_params_
print("Best parameters for Kernel Ridge Regression:", kernel_ridge_best_params)
print("")

###################################################################################################
#Cross validation plot for kernel ridge regression

kernel_ridge_scores = pd.DataFrame(kernel_ridge_grid.cv_results_)

plt.figure(figsize=(12, 8))
sns.heatmap(kernel_ridge_scores.pivot_table(index='param_alpha', columns='param_kernel', values='mean_test_score'), annot=True, cmap='viridis')
plt.title('Kernel Ridge Regression Cross-Validation Scores')
plt.xlabel('Kernel')
plt.ylabel('Alpha')
plt.show()

###################################################################################################

# Task 2.6: Compare Performance Across Different Models

# Define a function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    # Make predictions
    predictions = model.predict(X_test)
    # Calculate R2 score 
    ss_res = np.sum((y_test - predictions) ** 2)  
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)  
    r2 = 1 - (ss_res / ss_tot)  
    return r2 
# Define different models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=lasso_best_alpha, max_iter=1000000), 
    "Ridge Regression": Ridge(alpha=ridge_best_alpha, max_iter=38000),
    "Decision Tree": DecisionTreeRegressor(max_depth=tree_best_depth, random_state=42),
    "Adaboost": AdaBoostRegressor(random_state=42),
    "Kernel Ridge Regression": KernelRidge(alpha=kernel_ridge_best_params['alpha'], 
                                           kernel=kernel_ridge_best_params['kernel'], 
                                           gamma=kernel_ridge_best_params['gamma'], 
                                           degree=kernel_ridge_best_params['degree'])
}

# Define training set sizes to test
train_sizes = [0.9, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]

# Loop over models
for model_name, model in models.items():
    print("Model:", model_name)
    for train_size in train_sizes:
        # Split the data into training and testing sets with the specified size
        X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_scaled, y, test_size=(1-train_size), random_state=42)
        X_train_nonstd, X_test_nonstd, y_train_nonstd, y_test_nonstd = train_test_split(X, y, test_size=(1-train_size), random_state=42)

        # Evaluate model performance on standardized data
        r2_std = evaluate_model(model, X_train_std, X_test_std, y_train_std, y_test_std)  # <span style="color: red;">Updated</span>
        
        # Evaluate model performance on non-standardized data
        r2_nonstd = evaluate_model(model, X_train_nonstd, X_test_nonstd, y_train_nonstd, y_test_nonstd)  # <span style="color: red;">Updated</span>
        print("")
        print(f"Train Size: {train_size}, Standardized Data R2: {r2_std}, Non-Standardized Data R2: {r2_nonstd}")  # <span style="color: red;">Updated</span>
        print("")

###################################################################################################
#Residual plots

models = {
    "Linear Regression": linear_model,
    "Lasso Regression": lasso_best,
    "Ridge Regression": ridge_best
}

plt.figure(figsize=(18, 6))
for i, (model_name, model) in enumerate(models.items(), 1):
    predictions = model.predict(X_scaled)
    residuals = y - predictions
    
    plt.subplot(1, 3, i)
    plt.scatter(predictions, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'Residuals for {model_name}')

plt.tight_layout()
plt.show()







    