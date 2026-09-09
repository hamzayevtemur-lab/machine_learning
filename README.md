# Classical Machine Learning: Scratch Algorithms & Applications 🤖

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A structured machine learning repository featuring scratch implementations of core ML algorithms, hands-on homework exercises (Lessons 1–28), feature engineering pipelines, and mini web applications.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Algorithm Implementations](#-algorithm-implementations)
- [Homework Tasks & Curriculum](#-homework-tasks--curriculum)
- [Mini Applications](#-mini-applications)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## 💡 Overview

This repository covers classical machine learning principles:
* **Supervised Learning:** Linear Regression, Multiple Linear Regression, Polynomial Regression, Ridge & Lasso Regularization, Logistic Regression, Decision Trees, Support Vector Machines (SVM / SVR), Naive Bayes, $K$-Nearest Neighbors (KNN).
* **Unsupervised Learning:** $K$-Means Clustering, DBSCAN, Principal Component Analysis (PCA).
* **Evaluation & Diagnostics:** Cross-validation strategies, confusion matrices, precision/recall curves, elbow method.
* **Pipelines & Web Apps:** Scikit-Learn `Pipeline` integration, serialized model deployment (`.pkl`) with Flask microservices.

---

## 🛠️ Algorithm Implementations

Located in [`MyModels/`](MyModels/):

```python
# Scratch K-Nearest Neighbors Classification Example
import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = [self._predict_one(x) for x in np.array(X)]
        return np.array(predictions)

    def _predict_one(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return np.bincount(k_nearest_labels).argmax()
```

---

## 📚 Homework Tasks & Curriculum

The [`homework/`](homework/) directory contains assignments covering Lessons 1 through 28:

* **Lessons 1–5:** Exploratory Data Analysis, Preprocessing, Single & Multiple Linear Regression.
* **Lessons 6–10:** Gradient Descent optimization, L1/L2 Regularization (Ridge/Lasso), Logistic Regression, Binary Classification metrics.
* **Lessons 11–15:** Multi-class Classification, Naive Bayes Classifier, $K$-Nearest Neighbors.
* **Lessons 16–20:** Decision Trees (Classification & Regression), Random Forests, SVM & SVR.
* **Lessons 21–28:** $K$-Means Clustering, DBSCAN, PCA dimensionality reduction, Cross-Validation, Hyperparameter Tuning, Model Pipelines.

---

## 📱 Mini Applications

Located in [`practise_apps/`](practise_apps/):
* `digit_pred/`: Handwritten Digit Prediction web app using a trained classifier and Flask REST API.
* `car_price/`: Car price valuation model and predictor application.

---

## 📁 Repository Structure

```
.
├── datasets/                       # ML Datasets (car_price, housing)
├── homework/                       # Homework Assignments (Lessons 1 – 28)
├── models/                         # Serialized Models (.pkl)
├── MyModels/                       # Custom ML Algorithms Written from Scratch
│   ├── linear_regression.py
│   ├── multilinear_regression.py
│   ├── polynomial_regression.py
│   ├── logistic_regression.ipynb
│   ├── knn_classification.py
│   ├── knn_regression.py
│   ├── tree_classification.py
│   ├── tree_regression.py
│   ├── pca_implementation.py
│   └── kmeans.py
├── practise/                       # Interactive Jupyter Notebooks
├── practise_apps/                  # Mini Applications & Deployment
├── requirements.txt                # Dependencies
├── LICENSE                         # MIT License
└── README.md                       # Documentation
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hamzayevtemur-lab/machine_learning.git
   cd machine_learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run an algorithm script:
   ```bash
   python MyModels/knn_classification.py
   ```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
