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



# Keeping the original work safe. Email atharvasinnarkar@gmail.com for the file and mention the proper usecase.
