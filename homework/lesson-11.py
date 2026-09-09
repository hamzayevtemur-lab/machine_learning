import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load dataset
data=load_wine()

X=pd.DataFrame(data.data, columns=data.feature_names) #type:ignore
y=pd.Series(data.target) #type:ignore

#Display info
print(X.head())
print("Shape:", X.shape)

print("\nSummary statistics:")
print(X.describe())

print("\nClass distribution:")
print(y.value_counts())

###### Baseline Model (Train-Test Split)

#Split
X_train,X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Model
model=LogisticRegression(max_iter=5000)

#Train
model.fit(X_train, y_train)

#Predict
y_pred=model.predict(X_test)

#Accuracy
acc=accuracy_score(y_test, y_pred)
print("Baseline Accuracy:", acc)


##### K-Fold Cross Validation

kf=KFold(n_splits=5, shuffle=True, random_state=42)

model=LogisticRegression(max_iter=10000)

scores=cross_val_score(model, X, y, cv=kf, scoring="accuracy") #One of the important function

print("Fold scores:", scores)
print("Mean:", scores.mean())
print("STD:", scores.std())


######## Stratified K-Fold Cross Validation

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_model=LogisticRegression(max_iter=10000)
rf_model=RandomForestClassifier(random_state=42)

log_score=cross_val_score(log_model, X, y, cv=skf)
rf_scores=cross_val_score(rf_model, X, y, cv=skf)

print("Logistic Regression Mean:", log_score.mean())
print("Logistic Regression Std:", log_score.std())

print("Random Forest Mean:", rf_scores.mean())
print("Random Forest Std:", rf_scores.std())


####### Cross Validation with Multiple Metrics

scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]

log_results=cross_validate(log_model, X, y, cv=skf, scoring=scoring)
rf_results=cross_validate(rf_model, X, y, cv=skf, scoring=scoring)

def print_results(name, results):
    print(f"\n{name}")
    for metric in scoring:
        scores=results[f"test_{metric}"]
        print(f"{metric}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    

print_results("Logistic Regression", log_results)
print_results("Random Forest", rf_results)



######## Hyperparameter Tuning (GridSearchCV)

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 4, 6]
}

grid=GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Score:", grid.best_score_)
print("Best Params:", grid.best_params_)
print("Best Estimator:", grid.best_estimator_)


# Evaluate Best Model
best_model=grid.best_estimator_

y_pred=best_model.predict(X_test)

print("Test Accuracy (Tuned Model):", accuracy_score(y_test, y_pred))