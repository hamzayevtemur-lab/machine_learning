#!/usr/bin/env python3
"""
Converts all Machine Learning homework scripts (lesson-1.py through lesson-28.py)
into modular, cell-by-cell Jupyter Notebooks (.ipynb) inside `homework_ipynb/`.
Every code cell is preceded by an explanatory markdown cell describing the mathematical
concepts, algorithms, functions, and expected behaviors.
"""

import os
import re
import json

ML_DIR = "/Users/mac/Desktop/Machine Learning/ML"
HOMEWORK_DIR = os.path.join(ML_DIR, "homework")
OUTPUT_DIR = os.path.join(ML_DIR, "homework_ipynb")

os.makedirs(OUTPUT_DIR, exist_ok=True)

LESSON_METADATA = {
    "lesson-1": {
        "title": "Lesson 1: Exploratory Data Analysis & Pandas Essentials",
        "desc": "Foundational data exploration, handling missing values, statistical summarization (`describe()`, `info()`), correlation analysis, and data visualization using Matplotlib & Seaborn."
    },
    "lesson-2": {
        "title": "Lesson 2: End-to-End Dataset Preprocessing & Stratified Splitting",
        "desc": "California Housing dataset pipeline: Handling skewed numerical attributes, categorical encoding, stratified sampling based on income categories, and train/test partitioning."
    },
    "lesson-3": {
        "title": "Lesson 3: Simple Linear Regression from Scratch & Scikit-Learn",
        "desc": "Univariate Linear Regression: Fitting $y = w_1 x + w_0$. Derivation of Ordinary Least Squares (OLS), closed-form normal equation, residual sum of squares (RSS), and $R^2$ score evaluation."
    },
    "lesson-4": {
        "title": "Lesson 4: Multiple Linear Regression & Feature Scaling",
        "desc": "Multivariate Linear Regression: $y = X w + b$. Comparing Closed-Form Normal Equation $(X^T X)^{-1} X^T y$ with Gradient Descent. Analyzing collinearity and feature normalization (`StandardScaler`, `MinMaxScaler`)."
    },
    "lesson-5": {
        "title": "Lesson 5: Polynomial Regression & Bias-Variance Tradeoff",
        "desc": "Fitting non-linear relationships via `PolynomialFeatures`. Diagnosing Underfitting (High Bias) vs Overfitting (High Variance) through learning curves and validation loss."
    },
    "lesson-6": {
        "title": "Lesson 6: Gradient Descent Optimization: Batch, SGD & Mini-Batch",
        "desc": "Iterative parameter optimization: Batch Gradient Descent (exact gradient), Stochastic Gradient Descent (SGD - fast, noisy), and Mini-Batch GD (balanced vectorization and convergence stability)."
    },
    "lesson-7": {
        "title": "Lesson 7: Regularization: Ridge Regression (L2 Penalty)",
        "desc": "Controlling model complexity and multicollinearity with L2 Tikhonov regularization: $J(w) = \\text{MSE}(w) + \\alpha \\sum w_i^2$. Analytical solution and shrinking coefficient weights."
    },
    "lesson-8": {
        "title": "Lesson 8: Regularization: Lasso Regression (L1 Penalty) & Feature Selection",
        "desc": "Sparse parameter optimization with L1 regularization: $J(w) = \\text{MSE}(w) + \\alpha \\sum |w_i|$. Automatic feature selection via non-differentiable subgradient descent driving coefficients to exact zero."
    },
    "lesson-9": {
        "title": "Lesson 9: Logistic Regression from Scratch & Sigmoid Activation",
        "desc": "Binary classification foundations: Sigmoid activation $\\sigma(z) = \\frac{1}{1 + e^{-z}}$, Log-Loss (Binary Cross-Entropy), and gradient derivation $\\nabla J(w) = \\frac{1}{m} X^T (\\hat{y} - y)$."
    },
    "lesson-10": {
        "title": "Lesson 10: Logistic Regression Decision Boundaries & One-vs-Rest",
        "desc": "Mapping linear decision surfaces, probability thresholds, and extending binary logistic regression to multi-class problems using the One-vs-Rest (OvR) strategy."
    },
    "lesson-11": {
        "title": "Lesson 11: Classification Diagnostics & Evaluation Metrics",
        "desc": "Beyond accuracy: Confusion Matrix, Precision $\\frac{TP}{TP+FP}$, Recall $\\frac{TP}{TP+FN}$, F1-Score (Harmonic Mean), ROC Curves, and Area Under the Curve (AUC-ROC)."
    },
    "lesson-12": {
        "title": "Lesson 12: Softmax Regression & Multi-Class Cross-Entropy",
        "desc": "Multinomial Logistic Regression: Generalizing to $K$ classes using Softmax probabilities $P(y=k|x) = \\frac{e^{z_k}}{\\sum e^{z_j}}$ and Categorical Cross-Entropy Loss."
    },
    "lesson-13": {
        "title": "Lesson 13: Text Representation: Bag of Words & CountVectorizer",
        "desc": "Natural Language Preprocessing for ML: Tokenization, stopword filtering, n-gram extraction, and building term-document frequency matrices with `CountVectorizer`."
    },
    "lesson-14": {
        "title": "Lesson 14: Naive Bayes Classification (Gaussian & Multinomial)",
        "desc": "Probabilistic Bayesian inference: Bayes Theorem $P(C|X) = \\frac{P(X|C)P(C)}{P(X)}$ under conditional feature independence assumption. Laplace smoothing and text spam filtering."
    },
    "lesson-15": {
        "title": "Lesson 15: K-Nearest Neighbors (KNN) Classifier from Scratch",
        "desc": "Instance-based non-parametric learning: Euclidean ($L_2$) and Manhattan ($L_1$) distances, majority vote voting rule, boundary visualization, and selecting optimal $k$."
    },
    "lesson-16": {
        "title": "Lesson 16: KNN Regression & The Curse of Dimensionality",
        "desc": "Continuous target estimation via distance-weighted average $k$-nearest values. Analyzing the exponential data sparsity and distance metric breakdown in high dimensions."
    },
    "lesson-17": {
        "title": "Lesson 17: Decision Tree Fundamentals: Entropy & Gini Impurity",
        "desc": "Recursive binary splitting: Shannon Entropy $H(S) = -\\sum p_i \\log_2 p_i$, Gini Impurity $G(S) = 1 - \\sum p_i^2$, Information Gain maximization, and leaf stopping conditions."
    },
    "lesson-18": {
        "title": "Lesson 18: Decision Tree Classifier from Scratch & Cost-Complexity Pruning",
        "desc": "Full CART algorithm implementation: Finding best feature/threshold splits, recursive node generation, tree depth limits, and post-pruning via `ccp_alpha`."
    },
    "lesson-19": {
        "title": "Lesson 19: Decision Tree Regression & Variance Reduction",
        "desc": "Continuous CART trees: Splitting criterion based on Mean Squared Error (MSE) / Variance Reduction, step-wise piecewise constant predictions, and residual evaluation."
    },
    "lesson-20": {
        "title": "Lesson 20: Ensemble Methods: Bagging & Random Forests",
        "desc": "Bootstrap Aggregation (Bagging) and Random Forests: Decorrelating trees with random feature sub-sampling, Out-of-Bag (OOB) error estimation, and feature importance scores."
    },
    "lesson-21": {
        "title": "Lesson 21: Support Vector Machines (SVM): Hard/Soft Margin & Kernels",
        "desc": "Maximum Margin Hyperplane: Primal & Dual Lagrangian optimization, support vectors, soft-margin slack variables ($C$), and Kernel Trick (Linear, Polynomial, Radial Basis Function RBF)."
    },
    "lesson-24": {
        "title": "Lesson 24: Support Vector Regression (SVR) & Hyperparameter Grid Search",
        "desc": "$\epsilon$-insensitive Tube Regression: Tolerating errors within $\pm \epsilon$, penalizing outer deviations via slack variables, and exhaustive `GridSearchCV` hyperparameter tuning."
    },
    "lesson-25": {
        "title": "Lesson 25: Principal Component Analysis (PCA) from Scratch",
        "desc": "Unsupervised dimensionality reduction: Centering data, computing the Covariance Matrix $\Sigma = \\frac{1}{m} X^T X$, Eigenvalue Decomposition $\Sigma v = \lambda v$, explained variance ratio, and projection."
    },
    "lesson-26": {
        "title": "Lesson 26: K-Means Clustering from Scratch & Elbow Method",
        "desc": "Centroid-based unsupervised partitioning: Lloyd's algorithm (Assignment $\\to$ Update steps), convergence criteria, Inertia (Within-Cluster Sum of Squares), Elbow method, and Silhouette analysis."
    },
    "lesson-27": {
        "title": "Lesson 27: DBSCAN: Density-Based Spatial Clustering of Applications with Noise",
        "desc": "Density clustering: Core points, Border points, and Noise outliers based on radius $\epsilon$ and `min_samples`. Discovering arbitrary non-convex cluster shapes without specifying cluster count $k$."
    },
    "lesson-28": {
        "title": "Lesson 28: End-to-End Scikit-Learn Pipelines & Model Deployment",
        "desc": "Production-grade ML architectures: `ColumnTransformer` (numerical imputation/scaling + categorical one-hot encoding), `Pipeline`, nested Cross-Validation, model serialization (`pickle`/`joblib`), and REST deployment."
    }
}

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def split_py_into_blocks(code_str):
    """
    Intelligently splits python script into modular code blocks based on logical step comments.
    """
    lines = code_str.split("\n")
    blocks = []
    curr_header = ""
    curr_code = []

    def push():
        nonlocal curr_header, curr_code
        code_body = "\n".join(curr_code).strip()
        if code_body:
            blocks.append({
                "header": curr_header.strip(),
                "code": code_body
            })
        curr_header = ""
        curr_code = []

    for line in lines:
        stripped = line.strip()
        
        # Check for section markers like '# 1.', '### ...', '## ...'
        is_marker = bool(re.match(r'^(#{1,4}\s+[A-Za-z0-9]|#\s*\d+[\.\)]|\"\"\"|\'\'\')', stripped))
        is_func_or_class = bool(re.match(r'^(def |class )', line) and not line.startswith(" "))
        
        if (is_marker or is_func_or_class) and len(curr_code) >= 6:
            push()
            
        if is_marker and not curr_header:
            curr_header = re.sub(r'^[#\s]+', '', stripped)
        else:
            curr_code.append(line)

    push()
    return blocks

def generate_step_explanation(step_num, header, code_snippet):
    title = header if header else f"Step {step_num}: Implementation Block"
    exp = [f"### 🔹 {title}"]
    
    if "import " in code_snippet or "from " in code_snippet:
        exp.append("**Objective**: Import Core Numerical & Machine Learning Libraries.")
        exp.append("- Sets up `numpy` for linear algebra, `pandas` for dataframes, `matplotlib`/`seaborn` for plotting, and `sklearn` estimators.")
    elif "read_csv" in code_snippet or "load_" in code_snippet or "fetch_" in code_snippet or "make_blobs" in code_snippet or "make_moons" in code_snippet:
        exp.append("**Objective**: Dataset Loading & Synthetic Data Generation.")
        exp.append("- Ingests or synthesizes sample data. Inspects feature shapes, target distributions, and missing values.")
    elif "train_test_split" in code_snippet or "StandardScaler" in code_snippet or "MinMaxScaler" in code_snippet:
        exp.append("**Objective**: Data Splitting & Feature Normalization.")
        exp.append("- Splits data into independent train and test sets to evaluate generalization.")
        exp.append("- Scales numerical features to zero mean ($\mu=0$) and unit variance ($\sigma=1$) to prevent features with large scales from dominating gradients or distance metrics.")
    elif "class " in code_snippet:
        exp.append("**Objective**: Algorithm / Model Architecture Implementation from Scratch.")
        exp.append("- Encapsulates algorithm math (`fit()`, `predict()`, distance/loss computations) following Scikit-Learn's estimator interface.")
    elif ".fit(" in code_snippet:
        exp.append("**Objective**: Model Training & Parameter Optimization.")
        exp.append("- Executes optimization (OLS normal equation, gradient descent, tree split search, or centroid convergence) on the training set.")
    elif "predict(" in code_snippet or "predict_proba" in code_snippet:
        exp.append("**Objective**: Model Inference & Prediction Generation.")
        exp.append("- Computes predictions or predicted probability scores on unseen test samples.")
    elif "accuracy_score" in code_snippet or "mean_squared_error" in code_snippet or "confusion_matrix" in code_snippet or "r2_score" in code_snippet or "classification_report" in code_snippet:
        exp.append("**Objective**: Model Evaluation & Diagnostic Metrics.")
        exp.append("- Calculates quantitative metrics ($R^2$, MSE, Accuracy, Precision, Recall, F1) to measure generalization performance.")
    elif "plt." in code_snippet or "sns." in code_snippet:
        exp.append("**Objective**: Visual Diagnostics & Decision Boundary Plots.")
        exp.append("- Visualizes feature relationships, regression curves, decision surfaces, or clustering partitions.")
    else:
        exp.append("**Objective**: Execution & Utility Transformation.")
        exp.append("- Transforms data features, computes intermediate statistics, or runs diagnostic evaluations.")
        
    return "\n\n".join(exp)

py_files = sorted([f for f in os.listdir(HOMEWORK_DIR) if f.startswith("lesson-") and f.endswith(".py")])
print(f"Converting {len(py_files)} ML homework scripts into modular notebooks...")

for py_file in py_files:
    base_name = py_file.replace(".py", "")
    full_path = os.path.join(HOMEWORK_DIR, py_file)
    
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    blocks = split_py_into_blocks(content)
    
    # Fallback to evenly distributed chunks if few blocks detected
    if len(blocks) <= 2:
        lines = content.split("\n")
        chunk_sz = max(15, len(lines) // 4)
        blocks = []
        for c in range(0, len(lines), chunk_sz):
            chunk_code = "\n".join(lines[c:c+chunk_sz]).strip()
            if chunk_code:
                blocks.append({"header": f"Execution Step {len(blocks)+1}", "code": chunk_code})

    meta = LESSON_METADATA.get(base_name, {
        "title": f"Lesson: {base_name.upper()}",
        "desc": "Hands-on machine learning implementation and empirical analysis."
    })
    
    cells = [
        md_cell(f"# 📘 {meta['title']}\n\n{meta['desc']}\n\n---\n**Interactive Step-by-Step Notebook**: Execute cells sequentially to observe data flow, mathematical transformations, and model evaluations.")
    ]
    
    for idx, blk in enumerate(blocks, 1):
        explanation = generate_step_explanation(idx, blk["header"], blk["code"])
        cells.append(md_cell(explanation))
        cells.append(code_cell(blk["code"]))
        
    cells.append(md_cell("## 🎯 Summary & Key Takeaways\n"
                         "1. **Core Insight**: Review the printed parameters, loss curves, and evaluation metrics above.\n"
                         "2. **Best Practice**: Always ensure proper feature scaling, validation splitting, and metric selection tailored to the problem distribution.\n"
                         "3. **Next Steps**: Compare these results with related models in subsequent lessons."))

    nb = create_notebook(cells)
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {out_path} ({len(cells)} cells)")

# Copy datasets to homework_ipynb if available
datasets_dir = os.path.join(ML_DIR, "datasets")
if os.path.exists(datasets_dir):
    import shutil
    for d_file in os.listdir(datasets_dir):
        src = os.path.join(datasets_dir, d_file)
        dst = os.path.join(OUTPUT_DIR, d_file)
        if os.path.isfile(src):
            shutil.copy(src, dst)
            print(f"Copied dataset {d_file} to homework_ipynb")

print("All ML homework notebooks generated successfully!")
