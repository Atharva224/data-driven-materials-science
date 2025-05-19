
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




# Keeping the original work safe. Email atharvasinnarkar@gmail.com for the file and mention the proper usecase.
