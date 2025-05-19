
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





# Keeping the original work safe. Email atharvasinnarkar@gmail.com for the file and mention the proper usecase.
