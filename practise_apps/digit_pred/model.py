import numpy as np
import pandas as pd
import joblib
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# ── Load MNIST (full 70,000 samples) ─────────────────────────────────────────
print("Loading dataset...")
mnist = fetch_openml("mnist_784", version=1, as_frame=True)

X, y = mnist.data, mnist.target.astype(int)

print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")

# ── Split ──────────────────────────────
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ── Pipeline ────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=42))
])

# ── GridSearch ─────────────────────
# Wider C range + saga solver (faster on large data, supports L1)
param_grid = {
    "clf__C": [0.05, 0.1, 0.5, 1.0],
    "clf__solver": ["saga"],
    "clf__penalty": ["l2"],        
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1          # use all cores
)

# ── Train ─────────────────────────────
print("Starting GridSearch...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", f"{grid.best_score_:.4f}")
print("Test Accuracy:   ", f"{best_model.score(X_test, y_test):.4f}")

# ── Save ─────────────────────────────
joblib.dump(best_model, "model/digit_model.pkl")
joblib.dump(X.columns.tolist(), "model/columns.pkl")

print("\nModel saved to model/digit_model.pkl")
print("Columns saved to model/columns.pkl")