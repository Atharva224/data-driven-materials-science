# -*- coding: utf-8 -*-
"""
@author: Atharva Sinnarkar
"""
#Task 1.2

import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt 
from pca import PrincipalComponentAnalysis

if __name__ == "__main__":

    # Load data from CSV file
    X = read_csv('features-bulk.csv')
    '''
    print("size of feature dataset is:",X.shape)
    '''
    # Perform PCA
    pca = PrincipalComponentAnalysis(X.shape[1])
    eigenvalues, _ = pca.train(X)
    '''
    print(eigenvalues, _.shape)
    '''

    # Compute variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    '''
    print(explained_variance_ratio.shape)
    '''
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Plot cumulative spectrum of principal components
    plt.figure(figsize=(15, 5))

    # Plot contribution of each PCA
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(1, len(eigenvalues) + 1), eigenvalues / np.sum(eigenvalues))
    plt.xlabel('Principal Component')
    plt.ylabel('Contribution to Variance')
    plt.title('Contribution of each Principal Component')

    # Plot cumulative explained variance ratio
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Spectrum of Principle Components')
    plt.title('Cumulative Explained Variance Ratio')
    plt.tight_layout()
    plt.show()


    pass
    
    