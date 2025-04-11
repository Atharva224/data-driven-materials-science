# 🧠 Predicting the Bulk Modulus of Inorganic Crystals

This repository contains a data-driven machine learning pipeline developed for predicting the bulk modulus of inorganic crystals. The project was completed as part of the *Materials Informatics* module at FAU and explores PCA-based dimensionality reduction and various regression models. Techniques include Lasso, Ridge, Tree regressors, Kernel Ridge Regression, and feature selection methods.

---

## 📁 Contents


---

## 🧠 Key Concepts

- Dimensionality reduction using Principal Component Analysis (PCA)
- Linear models: Least Squares, Ridge, Lasso (with regularization tuning)
- Polynomial feature expansion with cross terms
- Tree-based regressors: Decision Trees, Adaboost
- Kernel Ridge Regression with RBF and polynomial kernels
- Recursive Feature Elimination (RFE) and Least Angle Regression (LARS)
- Evaluation metrics: \( R^2 \), MSE, RMSE, model stability with training size
- Feature importance ranking and physical interpretability

---

## 📌 Implemented Tasks

| Task        | Description                                                       |
|-------------|-------------------------------------------------------------------|
| Task 1.1     | Implement PCA class and validate eigen decomposition             |
| Task 1.2     | Plot PCA component spectrum and analyze variance capture         |
| Task 2.1     | Derive and implement Linear, Lasso, Ridge regressors             |
| Task 2.2     | Hyperparameter tuning via GridSearchCV                           |
| Task 2.3     | Polynomial feature expansion and model retraining                |
| Task 2.4     | Tree regressor with Adaboost + feature importance                |
| Task 2.5     | Kernel Ridge Regression with RBF/Polynomial kernels              |
| Task 2.6     | Compare model performance (standardized vs non-standardized)     |
| Task 3.1     | Least Angle Regression (LARS) for feature selection              |
| Task 3.2     | Recursive Feature Elimination with Lasso and Tree models         |
| Task 3.3     | Compare selected features from all models                        |

---

## 💻 Requirements

- Python 3.8+
- Libraries: `numpy`, `scikit-learn`, `matplotlib`, `pandas`

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib
