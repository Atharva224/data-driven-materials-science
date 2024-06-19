"""
@author: Atharva Sinnarkar
"""

#Task 1.1

import numpy as np


####################################################################################################

class PrincipalComponentAnalysis():
    """
    Principal Component Analysis (PCA) implementation.
    """
    def __init__(self, n_components):
        """
        Don't change anything in this function. This  function is not
        necessary, but is just used to tell you which variables you should use.
        We use these variables to check your solution.
        """
        self.n_components = n_components
        self.components = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.means = None

    def train(self, xtrain):
        """
        Calculate the PCA as explained in the script. Please use the variables
        mentioned above. This is techincally not necessary but we use it to
        check your solution and give you hints where something has gone wrong.

        xtrain: np.array of shape (nsamples,ndimensions)
        """
        # calculate class means
        self.means = np.mean(xtrain, axis=0)
        xtrain_centered = xtrain - self.means

        # set up covariance matrix
        covariance = np.cov(xtrain_centered.T)

        # calculate eigenvectors of covariance matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(covariance)

        # Sort eigenvalues and eigenvectors in descending order
        idxs = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idxs]
        self.eigenvectors = self.eigenvectors[:, idxs]

        # Store the principal components
        self.components = self.eigenvectors[:, :self.n_components]

        return self.eigenvalues, self.eigenvectors

    def transform(self, xtrain):
        """
        Transform data into from the original coordinate system to the
        coordinate system of the principal components.

        x: np.array of shape (nsamples,ndimensions)
        """
        xtrain_centered = xtrain - self.means
        return np.dot(xtrain_centered, self.components)

    def backtransform(self, x_transformed):
        """
        Transform data from the coordinate system of the principal components
        to the original coordinate system.
        """
        return np.dot(x_transformed, self.components.T) + self.means

if __name__ == "__main__":
    pass
    
