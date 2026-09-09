import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)

# Load Dataset
data=load_breast_cancer()

X=data.data #type:ignore
y=data.target #type:ignore

print("Feature shape", X.shape)
print("Target shape", X.shape)

# Train-Test Split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=y
)

# Scaling
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


#### Boosting (AdaBoost)
from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(
    n_estimators=50 ,
    learning_rate=0.1,
    random_state=42
)

ada.fit(X_train_scaled, y_train)

train_pred=ada.predict(X_train_scaled)
test_pred=ada.predict(X_test_scaled)

# Evaluation
print("=== AdaBoost ===")

print("Train Accuracy:", accuracy_score(y_train, train_pred))

print("Test Accuracy:", accuracy_score(y_test, test_pred))

print("Train F1:", f1_score(y_train, train_pred))

print("Test F1:", f1_score(y_test, test_pred))


#### Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

# Experiment 1
gb_1=GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    random_state=42
)

gb_1.fit(X_train_scaled, y_train)

pred_1=gb_1.predict(X_test_scaled)

acc_1=accuracy_score(y_test, pred_1)
f1_1=f1_score(y_test, pred_1)

# Experiment 2
gb_2=GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)

gb_2.fit(X_train_scaled, y_train)

pred_2=gb_2.predict(X_test_scaled)

acc_2=accuracy_score(y_test, pred_2)
f1_2=f1_score(y_test, pred_2)

# Experiment 3
gb_3=GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.01,
    random_state=42
)

gb_3.fit(X_train_scaled, y_train)

pred_3=gb_3.predict(X_test_scaled)

acc_3=accuracy_score(y_test, pred_3)
f1_3=f1_score(y_test, pred_3)

# Result Table
results=pd.DataFrame({
    "n_estimators":[50, 100, 200],
    "learning_rate":[0.1, 0.05, 0.01],
    "Accuracy":[acc_1, acc_2, acc_3],
    "F1-score": [f1_1, f1_2, f1_3]
})

print(results)

### GridSearchCV
from sklearn.model_selection import GridSearchCV

gb_model=GradientBoostingClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [1, 2, 3]
}

grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# Best Parameters
print("Best Parameters:")
print(grid_search.best_params_)

# Best Score
print("Best CV Score:")
print(grid_search.best_score_)

# Best Model
best_model=grid_search.best_estimator_

# Evaluation
pred_grid=best_model.predict(X_test_scaled)

print("Best Model Test Accuracy:", accuracy_score(y_test, pred_grid))

print("Best Model F1 Score:", f1_score(y_test, pred_grid))


########## Stacking Classifier 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import StackingClassifier

# Base Models
base_models=[
    ("lr", LogisticRegression(max_iter=5000)),
    ("dt", DecisionTreeClassifier(random_state=42)),
    ("knn", KNeighborsClassifier())
]

# Meta learner
meta_model=LogisticRegression(max_iter=5000)

# Build Stacking Model
stack_model=StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

stack_model.fit(X_train_scaled, y_train)

# Predictions
stack_pred=stack_model.predict(X_test_scaled)
stack_prob = stack_model.predict_proba(X_test_scaled)[:, 1] # type:ignore

# Evaluation
print("=== Stacking Classifier ===")

print("Accuracy:", accuracy_score(y_test, stack_pred))

print("ROC-AUC:", roc_auc_score(y_test, stack_prob))


