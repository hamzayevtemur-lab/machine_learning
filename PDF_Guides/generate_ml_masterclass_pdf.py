#!/usr/bin/env python3
"""
Generates the Ultimate Classical Machine Learning Masterclass PDF:
Classical_Machine_Learning_Complete_Masterclass.pdf
Covers ALL 28 Lessons from scratch implementations of Linear Regression, Ridge, Lasso,
Logistic Regression, Softmax, Naive Bayes, KNN, Decision Trees, Random Forests, SVM, SVR,
K-Means, DBSCAN, PCA, Pipelines, and Deployment.
Formatted with clean code cells, mathematical derivations, callout boxes, and an API encyclopedia.
"""

import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas

class MLNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(MLNumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(MLNumberedCanvas, self).showPage()
        super(MLNumberedCanvas, self).save()

    def draw_header_footer(self, page_count):
        self.saveState()
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#334155"))
            self.drawString(45, 752, "CLASSICAL MACHINE LEARNING: COMPLETE MASTER ENCYCLOPEDIA")
            self.drawRightString(567, 752, "COURSE MASTERCLASS • LESSONS 1–28")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(45, 744, 567, 744)

        # Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(45, 32, "Supervised • Unsupervised • Scratch Algorithms • Optimization • Pipelines")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(567, 32, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(45, 42, 567, 42)
        
        self.restoreState()


def build_ml_pdf(filename="Classical_Machine_Learning_Complete_Masterclass.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=50,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    C_PRIMARY   = colors.HexColor("#0F172A") # Slate 900
    C_ACCENT    = colors.HexColor("#0284C7") # Sky 600
    C_INDIGO    = colors.HexColor("#4338CA") # Indigo 700
    C_EMERALD   = colors.HexColor("#059669") # Emerald 600
    C_DARK      = colors.HexColor("#1E293B") # Slate 800
    C_TEXT      = colors.HexColor("#334155") # Slate 700
    C_BG_CODE   = colors.HexColor("#F8FAFC") # Slate 50
    C_BORDER    = colors.HexColor("#E2E8F0") # Slate 200
    C_CALLOUT_BG= colors.HexColor("#F0FDF4") # Emerald 50
    C_CALLOUT_BD= colors.HexColor("#10B981") # Emerald 500

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=C_PRIMARY,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13.5,
        textColor=C_ACCENT,
        spaceAfter=10
    )

    ch_style = ParagraphStyle(
        'ChapterHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=C_PRIMARY,
        spaceBefore=11,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=C_INDIGO,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.8,
        textColor=C_TEXT,
        spaceAfter=3.5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=C_TEXT,
        leftIndent=10,
        spaceAfter=2
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.6,
        leading=8.6,
        textColor=colors.HexColor("#0F172A")
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.6,
        leading=10.2,
        textColor=colors.HexColor("#065F46")
    )

    tbl_header_style = ParagraphStyle(
        'TblHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.white
    )

    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.0,
        leading=9.0,
        textColor=C_TEXT
    )

    tbl_cell_code_style = ParagraphStyle(
        'TblCellCode',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=6.6,
        leading=8.6,
        textColor=colors.HexColor("#0369A1")
    )

    story = []

    def code_box(code_text):
        p = Preformatted(code_text.strip(), code_style)
        t = Table([[p]], colWidths=[522])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_BG_CODE),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 3.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    def callout_box(text, title="Core Principle"):
        full_text = f"<b>{title}:</b> {text}"
        p = Paragraph(full_text, callout_style)
        t = Table([[p]], colWidths=[522])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_CALLOUT_BG),
            ('LINELEFT', (0,0), (0,-1), 3, C_CALLOUT_BD),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#A7F3D0")),
            ('TOPPADDING', (0,0), (-1,-1), 3.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ]))
        return t

    # ─────────────────────────────────────────────────────────────────────────
    # HEADER & TITLE BLOCK
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("📊 Classical Machine Learning: The Complete Masterclass", title_style))
    story.append(Paragraph("<b>Comprehensive Theory, Mathematics, Scratch Code & Scikit-Learn Reference:</b> Lessons 1 through 28", subtitle_style))
    
    meta_html = "<b>Curriculum Coverage:</b> Linear/Polynomial Regression &bull; Gradient Descent &bull; Regularization (Ridge/Lasso) &bull; Logistic & Softmax &bull; Naive Bayes &bull; KNN &bull; Decision Trees &bull; Random Forests &bull; SVM & SVR &bull; K-Means & DBSCAN &bull; PCA &bull; Pipelines & REST Deployment"
    meta_p = Paragraph(meta_html, ParagraphStyle('MetaText', parent=styles['Normal'], fontName='Helvetica', fontSize=7.0, leading=9.2, textColor=colors.HexColor("#475569")))
    meta_table = Table([[meta_p]], colWidths=[522])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 1: LINEAR REGRESSION & OPTIMIZATION (LESSONS 1–6)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 1: Linear Regression & Optimization from Scratch (Lessons 1–6)", ch_style))
    story.append(Paragraph(
        "Linear regression models the relationship between dependent target $y \\in \\mathbb{R}$ and feature vector $x \\in \\mathbb{R}^d$ as an affine combination: $\\hat{y} = w^T x + b = X w$.",
        body_style
    ))
    
    story.append(Paragraph("<b>1. Closed-Form Normal Equation vs. Gradient Descent:</b>", h2_style))
    story.append(Paragraph(
        "&bull; <b>Ordinary Least Squares (OLS) Loss:</b> $J(w) = \\frac{1}{2m} \\sum_{i=1}^m (\\hat{y}^{(i)} - y^{(i)})^2 = \\frac{1}{2m} \\|X w - y\\|^2$.<br/>"
        "&bull; <b>Normal Equation:</b> Setting $\\nabla J(w) = 0 \\implies w^* = (X^T X)^{-1} X^T y$ ($O(d^3)$ complexity, exact analytic solution).<br/>"
        "&bull; <b>Batch Gradient Descent Update:</b> $w := w - \\eta \\nabla J(w) = w - \\frac{\\eta}{m} X^T (X w - y)$ ($O(m \\cdot d)$ per step).",
        bullet_style
    ))
    
    lin_code = (
        "# Lesson 3 & 4: Linear Regression from Scratch (Normal Equation & Gradient Descent)\n"
        "import numpy as np\n\n"
        "class LinearRegressionScratch:\n"
        "    def __init__(self, lr=0.01, n_iters=1000, method='normal_equation'):\n"
        "        self.lr, self.n_iters, self.method = lr, n_iters, method\n"
        "        self.weights, self.bias = None, None\n\n"
        "    def fit(self, X, y):\n"
        "        m, d = X.shape\n"
        "        if self.method == 'normal_equation':\n"
        "            X_b = np.c_[np.ones((m, 1)), X] # Add bias intercept column\n"
        "            theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y\n"
        "            self.bias, self.weights = theta_best[0], theta_best[1:]\n"
        "        else: # Gradient Descent\n"
        "            self.weights, self.bias = np.zeros(d), 0.0\n"
        "            for _ in range(self.n_iters):\n"
        "                y_pred = X @ self.weights + self.bias\n"
        "                error = y_pred - y\n"
        "                self.weights -= self.lr * (1/m) * (X.T @ error)\n"
        "                self.bias -= self.lr * (1/m) * np.sum(error)\n\n"
        "    def predict(self, X):\n"
        "        return X @ self.weights + self.bias"
    )
    story.append(code_box(lin_code))
    story.append(Spacer(1, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 2: REGULARIZATION: RIDGE, LASSO & ELASTICNET (LESSONS 7–8)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 2: Regularization: Ridge (L2), Lasso (L1) & ElasticNet (Lessons 7–8)", ch_style))
    story.append(Paragraph(
        "Regularization penalizes large weight magnitudes to mitigate multicollinearity and overfitting (variance reduction):",
        body_style
    ))
    
    reg_diagram = (
        "                    REGULARIZATION OBJECTIVE FORMULATIONS\n\n"
        "   1. Ridge (L2 / Tikhonov):  J(w) = MSE(w) + alpha * sum(w_j^2)    ==>  w = (X^T X + alpha*I)^(-1) X^T y\n"
        "      • Shrinks coefficients smoothly toward zero; never sets them to exact 0.\n\n"
        "   2. Lasso (L1 / Sparsity):  J(w) = MSE(w) + alpha * sum(|w_j|)    ==>  Diamond constraint surface\n"
        "      • Drives uninformative feature weights to EXACT ZERO (automatic feature selection!).\n\n"
        "   3. ElasticNet (Hybrid):    J(w) = MSE(w) + r * alpha * L1 + ((1-r)/2) * alpha * L2"
    )
    story.append(code_box(reg_diagram))
    story.append(Spacer(1, 4))
    
    story.append(callout_box(
        "<b>When to use which?</b> Use <b>Ridge</b> as the default when features are dense and correlated. Use <b>Lasso</b> when expecting sparse signals with many irrelevant noise features. Use <b>ElasticNet</b> when features are highly correlated and $p > n$.",
        "Regularization Selection Guide"
    ))
    story.append(Spacer(1, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 3: LOGISTIC & MULTICLASS CLASSIFICATION (LESSONS 9–12)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 3: Logistic Regression, Softmax & Evaluation Metrics (Lessons 9–12)", ch_style))
    story.append(Paragraph(
        "Logistic regression maps linear projections to calibrated probabilities via the Sigmoid activation $\\sigma(z) = \\frac{1}{1 + e^{-z}}$. "
        "Training minimizes the Log-Loss (Binary Cross-Entropy):",
        body_style
    ))
    
    log_code = (
        "# Lessons 9 & 10: Binary Logistic Regression from Scratch\n"
        "class LogisticRegressionScratch:\n"
        "    def __init__(self, lr=0.01, n_iters=1000):\n"
        "        self.lr, self.n_iters = lr, n_iters\n"
        "        self.weights, self.bias = None, None\n\n"
        "    def _sigmoid(self, z):\n"
        "        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))\n\n"
        "    def fit(self, X, y):\n"
        "        m, d = X.shape\n"
        "        self.weights, self.bias = np.zeros(d), 0.0\n"
        "        for _ in range(self.n_iters):\n"
        "            probs = self._sigmoid(X @ self.weights + self.bias)\n"
        "            d_w = (1/m) * (X.T @ (probs - y))\n"
        "            d_b = (1/m) * np.sum(probs - y)\n"
        "            self.weights -= self.lr * d_w\n"
        "            self.bias -= self.lr * d_b\n\n"
        "    def predict_proba(self, X):\n"
        "        return self._sigmoid(X @ self.weights + self.bias)\n\n"
        "    def predict(self, X, threshold=0.5):\n"
        "        return (self.predict_proba(X) >= threshold).astype(int)"
    )
    story.append(code_box(log_code))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Comprehensive Classification Diagnostics (Lesson 11):</b>", h2_style))
    story.append(Paragraph(
        "&bull; <b>Precision:</b> $\\frac{TP}{TP + FP}$ (Accuracy of positive predictions; crucial when False Positives are costly, e.g. spam filters).<br/>"
        "&bull; <b>Recall (Sensitivity):</b> $\\frac{TP}{TP + FN}$ (Fraction of actual positives detected; crucial for medical diagnoses).<br/>"
        "&bull; <b>F1-Score:</b> Harmonic mean $2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ balancing both metrics.<br/>"
        "&bull; <b>ROC-AUC:</b> Plots True Positive Rate ($TPR = \\text{Recall}$) vs False Positive Rate ($FPR = \\frac{FP}{TN + FP}$) across all decision thresholds.",
        bullet_style
    ))
    story.append(Spacer(1, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 4: NAIVE BAYES & K-NEAREST NEIGHBORS (LESSONS 13–16)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 4: Naive Bayes & K-Nearest Neighbors (KNN) (Lessons 13–16)", ch_style))
    story.append(Paragraph(
        "<b>Naive Bayes</b> applies Bayes' Theorem under the conditional independence assumption $P(x_1, \\dots, x_d | C_k) = \\prod_{j=1}^d P(x_j | C_k)$. "
        "<b>K-Nearest Neighbors (KNN)</b> is a non-parametric instance-based learner assigning majority class labels based on distance metrics ($L_2$ Euclidean / $L_1$ Manhattan):",
        body_style
    ))
    
    knn_code = (
        "# Lesson 15: KNN Classifier from Scratch\n"
        "class KNNClassifierScratch:\n"
        "    def __init__(self, k=3, metric='euclidean'):\n"
        "        self.k, self.metric = k, metric\n"
        "    def fit(self, X, y):\n"
        "        self.X_train, self.y_train = np.array(X), np.array(y)\n"
        "    def predict(self, X):\n"
        "        X = np.array(X)\n"
        "        return np.array([self._predict_sample(x) for x in X])\n"
        "    def _predict_sample(self, x):\n"
        "        if self.metric == 'euclidean':\n"
        "            distances = np.linalg.norm(self.X_train - x, axis=1)\n"
        "        else: # Manhattan\n"
        "            distances = np.sum(np.abs(self.X_train - x), axis=1)\n"
        "        k_indices = np.argsort(distances)[:self.k]\n"
        "        k_labels = self.y_train[k_indices]\n"
        "        return np.bincount(k_labels).argmax()"
    )
    story.append(code_box(knn_code))
    story.append(Spacer(1, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 5: DECISION TREES & RANDOM FORESTS (LESSONS 17–20)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 5: Decision Trees & Random Forest Ensembles (Lessons 17–20)", ch_style))
    story.append(Paragraph(
        "Decision Trees perform recursive binary axis-aligned partition search to maximize impurity reduction (Information Gain):",
        body_style
    ))
    
    tree_math = (
        "                    SPLITTING CRITERIA FORMULATIONS\n\n"
        "   1. Shannon Entropy:       H(S) = - sum_{k=1}^K p_k * log_2(p_k)\n"
        "   2. Gini Impurity (CART):   G(S) = 1 - sum_{k=1}^K p_k^2\n"
        "   3. Information Gain:      IG(S, A) = Impurity(S) - sum_{v} (|S_v| / |S|) * Impurity(S_v)\n"
        "   4. Regression MSE Split:  Var(S) = (1/|S|) * sum_{i in S} (y_i - y_bar)^2"
    )
    story.append(code_box(tree_math))
    story.append(Spacer(1, 4))
    
    rf_code = (
        "# Lesson 20: Random Forest Classifier\n"
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        "rf = RandomForestClassifier(\n"
        "    n_estimators=100,        # Number of de-correlated decision trees\n"
        "    max_depth=10,            # Maximum tree depth limit to prevent overfitting\n"
        "    max_features='sqrt',     # Random feature subsampling at each split (sqrt(d))\n"
        "    bootstrap=True,          # Sample with replacement\n"
        "    oob_score=True,          # Out-of-Bag validation evaluation\n"
        "    random_state=42\n"
        ")\n"
        "rf.fit(X_train, y_train)\n"
        "# Feature Importances based on mean impurity decrease:\n"
        "importances = rf.feature_importances_"
    )
    story.append(code_box(rf_code))
    story.append(Spacer(1, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 6: SUPPORT VECTOR MACHINES (SVM & SVR) (LESSONS 21 & 24)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 6: Support Vector Machines & SVR (Lessons 21 & 24)", ch_style))
    story.append(Paragraph(
        "SVM finds the maximum-margin hyperplane separating classes while tolerating margin violations via slack variables $\\xi_i$ (controlled by penalty $C$):",
        body_style
    ))
    
    svm_code = (
        "# Lessons 21 & 24: Support Vector Classifier & Regressor with Kernels\n"
        "from sklearn.svm import SVC, SVR\n\n"
        "# 1. Non-linear RBF Kernel Classifier: K(x, x') = exp(-gamma * ||x - x'||^2)\n"
        "svm_clf = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)\n"
        "svm_clf.fit(X_train, y_train)\n\n"
        "# 2. Epsilon-Insensitive Support Vector Regressor (SVR)\n"
        "svr_reg = SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')\n"
        "svr_reg.fit(X_train, y_train)\n"
        "# Predictions inside the +/- epsilon tube incur zero loss!"
    )
    story.append(code_box(svm_code))
    story.append(Spacer(1, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 7: UNSUPERVISED LEARNING & PCA (LESSONS 25–27)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 7: Clustering (K-Means, DBSCAN) & PCA (Lessons 25–27)", ch_style))
    story.append(Paragraph(
        "Unsupervised algorithms uncover latent clusters and low-dimensional manifolds without target labels:",
        body_style
    ))
    
    pca_code = (
        "# Lesson 25: Principal Component Analysis (PCA) from Scratch\n"
        "class PCAScratch:\n"
        "    def __init__(self, n_components=2):\n"
        "        self.n_components = n_components\n"
        "        self.components, self.mean, self.explained_variance_ratio = None, None, None\n\n"
        "    def fit(self, X):\n"
        "        self.mean = np.mean(X, axis=0)\n"
        "        X_centered = X - self.mean\n"
        "        # 1. Compute Covariance Matrix: Sigma = (1/m) * X^T X\n"
        "        cov_matrix = np.cov(X_centered, rowvar=False)\n"
        "        # 2. Eigenvalue Decomposition: Sigma * v = lambda * v\n"
        "        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)\n"
        "        # 3. Sort eigenvectors by descending eigenvalue magnitude\n"
        "        idx = np.argsort(eigenvalues)[::-1]\n"
        "        self.components = eigenvectors[:, idx[:self.n_components]]\n"
        "        self.explained_variance_ratio = eigenvalues[idx[:self.n_components]] / np.sum(eigenvalues)\n\n"
        "    def transform(self, X):\n"
        "        return (X - self.mean) @ self.components"
    )
    story.append(code_box(pca_code))
    story.append(Spacer(1, 4))
    
    clust_code = (
        "# Lessons 26 & 27: K-Means & DBSCAN Clustering\n"
        "from sklearn.cluster import KMeans, DBSCAN\n\n"
        "# K-Means: Minimizes Within-Cluster Sum of Squares (Inertia)\n"
        "kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)\n"
        "cluster_labels = kmeans.fit_predict(X)\n\n"
        "# DBSCAN: Density clustering discovering arbitrary shapes without pre-specifying k\n"
        "dbscan = DBSCAN(eps=0.3, min_samples=5)\n"
        "db_labels = dbscan.fit_predict(X) # -1 indicates noise outliers!"
    )
    story.append(code_box(clust_code))
    story.append(Spacer(1, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 8: PIPELINES, CROSS-VALIDATION & DEPLOYMENT (LESSON 28)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 8: Scikit-Learn Pipelines & Model Deployment (Lesson 28)", ch_style))
    story.append(Paragraph(
        "Production ML avoids data leakage by encapsulating feature imputation, categorical encoding, scaling, and estimators into a single callable `Pipeline`:",
        body_style
    ))
    
    pipe_code = (
        "# Lesson 28: Production-Grade ColumnTransformer & Pipeline\n"
        "import joblib\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
        "from sklearn.ensemble import RandomForestRegressor\n"
        "from sklearn.model_selection import GridSearchCV\n\n"
        "num_features = ['age', 'income', 'experience']\n"
        "cat_features = ['education', 'city']\n\n"
        "num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])\n"
        "cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])\n\n"
        "preprocessor = ColumnTransformer([\n"
        "    ('num', num_pipeline, num_features),\n"
        "    ('cat', cat_pipeline, cat_features)\n"
        "])\n\n"
        "full_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))])\n"
        "full_pipeline.fit(X_train, y_train)\n\n"
        "# Serialize complete end-to-end model for production deployment:\n"
        "joblib.dump(full_pipeline, 'production_model.pkl')\n"
        "# In production Flask app: model = joblib.load('production_model.pkl'); pred = model.predict(json_input)"
    )
    story.append(code_box(pipe_code))
    story.append(Spacer(1, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 9: MASTER CLASSICAL ML FUNCTION & API ENCYCLOPEDIA
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 9: Master Machine Learning Function & API Encyclopedia", ch_style))
    story.append(Paragraph(
        "Comprehensive reference table covering all critical Scikit-Learn estimators, transformers, metrics, and hyperparameters:",
        body_style
    ))
    
    encyclopedia_data = [
        [Paragraph("<b>Estimator / Function</b>", tbl_header_style), Paragraph("<b>Module</b>", tbl_header_style), Paragraph("<b>Exact Purpose &amp; Critical Hyperparameters</b>", tbl_header_style)],
        [Paragraph("<code>LinearRegression()</code>", tbl_cell_code_style), Paragraph("sklearn.linear_model", tbl_cell_style), Paragraph("Ordinary Least Squares regression via SVD / Normal Equation. Parameter: <code>fit_intercept</code>.", tbl_cell_style)],
        [Paragraph("<code>Ridge(alpha=1.0)</code>", tbl_cell_code_style), Paragraph("sklearn.linear_model", tbl_cell_style), Paragraph("L2 regularized linear regression. Prevents multicollinearity by shrinking weights smoothly.", tbl_cell_style)],
        [Paragraph("<code>Lasso(alpha=1.0)</code>", tbl_cell_code_style), Paragraph("sklearn.linear_model", tbl_cell_style), Paragraph("L1 regularized regression via coordinate descent. Sets uninformative weights to exact 0.", tbl_cell_style)],
        [Paragraph("<code>LogisticRegression()</code>", tbl_cell_code_style), Paragraph("sklearn.linear_model", tbl_cell_style), Paragraph("Log-Loss classification. Hyperparameters: <code>C</code> (inverse regularization), <code>penalty</code> (l1/l2), <code>multi_class</code>.", tbl_cell_style)],
        [Paragraph("<code>KNeighborsClassifier(k)</code>", tbl_cell_code_style), Paragraph("sklearn.neighbors", tbl_cell_style), Paragraph("Instance-based majority vote classification. Hyperparameters: <code>n_neighbors</code>, <code>metric</code> (minkowski).", tbl_cell_style)],
        [Paragraph("<code>DecisionTreeClassifier()</code>", tbl_cell_code_style), Paragraph("sklearn.tree", tbl_cell_style), Paragraph("Recursive partitioning. Parameters: <code>criterion</code> (gini/entropy), <code>max_depth</code>, <code>ccp_alpha</code>.", tbl_cell_style)],
        [Paragraph("<code>RandomForestClassifier()</code>", tbl_cell_code_style), Paragraph("sklearn.ensemble", tbl_cell_style), Paragraph("Bagging ensemble of de-correlated trees. Parameters: <code>n_estimators</code>, <code>max_features</code>, <code>oob_score</code>.", tbl_cell_style)],
        [Paragraph("<code>SVC(C, kernel)</code>", tbl_cell_code_style), Paragraph("sklearn.svm", tbl_cell_style), Paragraph("Maximum-margin classification with kernels (<code>linear</code>, <code>rbf</code>, <code>poly</code>), slack <code>C</code>, and <code>gamma</code>.", tbl_cell_style)],
        [Paragraph("<code>KMeans(n_clusters)</code>", tbl_cell_code_style), Paragraph("sklearn.cluster", tbl_cell_style), Paragraph("Centroid clustering. Parameters: <code>n_clusters</code>, <code>init='k-means++'</code>, <code>n_init</code>. Measures <code>inertia_</code>.", tbl_cell_style)],
        [Paragraph("<code>DBSCAN(eps, min_samples)</code>", tbl_cell_code_style), Paragraph("sklearn.cluster", tbl_cell_style), Paragraph("Density clustering discovering arbitrary non-convex clusters and isolating -1 noise outliers.", tbl_cell_style)],
        [Paragraph("<code>PCA(n_components)</code>", tbl_cell_code_style), Paragraph("sklearn.decomposition", tbl_cell_style), Paragraph("Eigenvalue decomposition of covariance matrix. Measures <code>explained_variance_ratio_</code>.", tbl_cell_style)],
        [Paragraph("<code>ColumnTransformer</code>", tbl_cell_code_style), Paragraph("sklearn.compose", tbl_cell_style), Paragraph("Applies distinct transformation pipelines to categorical and numerical feature subsets in parallel.", tbl_cell_style)],
        [Paragraph("<code>Pipeline(steps)</code>", tbl_cell_code_style), Paragraph("sklearn.pipeline", tbl_cell_style), Paragraph("Chains transformers and estimators sequentially to eliminate data leakage during cross-validation.", tbl_cell_style)],
        [Paragraph("<code>GridSearchCV</code>", tbl_cell_code_style), Paragraph("sklearn.model_selection", tbl_cell_style), Paragraph("Exhaustive hyperparameter tuning with K-fold cross-validation across a specified parameter grid.", tbl_cell_style)],
    ]
    
    ency_tbl = Table(encyclopedia_data, colWidths=[140, 95, 287])
    ency_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(ency_tbl)
    
    doc.build(story, canvasmaker=MLNumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_ml_pdf()
