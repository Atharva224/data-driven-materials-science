# Predicting the Bulk Modulus of Inorganic Crystals

This project focuses on predicting the bulk modulus of inorganic crystals using machine learning techniques applied to crystal composition data. The dataset utilized is sourced from Matminer, initially comprising crystal compositions which are further enriched using the "Magpie" feature generator. The primary objective is to predict the bulk modulus, a key mechanical property of materials.

Overview of the Project Structure:

1. Introduction:
Provides motivation and context for the project, outlining the significance of predicting material properties like bulk modulus for applications in material science and engineering.

2. Background and Theory:  
Discusses the theoretical foundations underlying the machine learning models employed, including:
Principal Component Analysis (PCA) for dimensionality reduction.
Linear regression models (Least Squares, Lasso, Ridge) and their regularization techniques.
Decision tree regressors and their modification methods (e.g., Adaboost, Gradient-Boosting).
Kernel Ridge Regression and its suitability for feature selection.
Feature selection methods such as Least Angle Regression (LARS) and Recursive Feature Elimination (RFE).

3. Methods:
Details the methodologies employed in the project:
PCA implementation for feature reduction.
Training and evaluation of various regression models (linear, tree-based, kernel-based).
Parameter optimization using techniques like Gridsearch.
Feature selection strategies to identify the most influential predictors of bulk modulus.

4. Discussion and Results:
Analyzes the results obtained from different models and techniques:
Comparison of model performances using metrics like R-squared.
Examination of feature importance across models and methods.
Insights into how model performance and feature selection are affected by data standardization and training set size.

5. Conclusion:
Summarizes the findings and their implications for predicting bulk modulus based on inorganic crystal compositions.
Discusses potential future directions for improving model accuracy and extending the analysis to other material properties.
Code and Implementation:
The project includes structured Python scripts and modules corresponding to each task outlined in the report. Each script is designed to be modular, with clear documentation and adherence to specified coding standards. Importantly, the provided functions and packages ensure reproducibility and compatibility with project requirements.

By following this structured approach, the project aims to contribute insights into the predictive modeling of material properties using machine learning, leveraging advanced techniques tailored to inorganic crystal data.
