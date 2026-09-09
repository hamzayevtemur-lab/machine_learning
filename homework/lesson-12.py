import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#Load dataset
data=load_breast_cancer()

X=data.data #type:ignore
y=data.target #type:ignore

#1.Train-test Split (80/20)
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)


#2. Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


#3. Train model
model=LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

#4. Predict and evaluate
y_pred=model.predict(X_test_scaled)
accuracy_baseline=accuracy_score(y_test, y_pred)

print("Baseline Accuracy:", accuracy_baseline)

############ Pipeline version #############

#Create pipeline
pipeline=Pipeline([
    ("scaler", StandardScaler()),
    ("model",LogisticRegression(max_iter=10000))
])

#Train
pipeline.fit(X_train, y_train)

#Predict
y_pred_pipe=pipeline.predict(X_test)

#Evaluate
accuracy_pipeline=accuracy_score(y_test, y_pred_pipe)

print("Pipeline Accuracy:", accuracy_pipeline)


############ GridSearchCV 

#Parameter grid
param_grid={
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty":["l2"],
    "model__solver":["lbfgs"]
}

#GridSearch
grid=GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

#Train
grid.fit(X_train, y_train)

#Best model
best_model=grid.best_estimator_

#Evaluate
y_pred=best_model.predict(X_test)

print("Best Params:", grid.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))