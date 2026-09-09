import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df=pd.read_csv("/Users/mac/Desktop/Machine Learning/ai-roadmap/content/ml/supervised/10 random forest/data/winequality-red.csv", sep=",")

# Preview
print("First 5 rows:", df.head())

#Info
print("info:",df.info())

# Missing values
print("Missing Values:", df.isnull().sum())

# Distribution of Features
df.hist(figsize=(12,10))
plt.show()


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X=df.drop("quality", axis=1)
y=df["quality"]

# Split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



# Train Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rf=RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred=rf.predict(X_test)

# Metrics
mae=mean_absolute_error(y_test, y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)



# Feature Importance
importances=rf.feature_importances_
features=X.columns

# Sort
indices=np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=45) #type:ignore
plt.show()


# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid={
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid=GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=1
)

grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)

# Evaluate Tuned Model

best_rf=grid.best_estimator_
y_pred_best=best_rf.predict(X_test)

print("Tuned R2:",r2_score(y_test, y_pred_best))



# Bonus: Classification Version
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def categorize(q):
    if q<=4:
        return "Low"
    elif q<=6:
        return "Medium"
    else:
        return "High"
    

df["quality_label"]=df["quality"].apply(categorize)

X=df.drop(["quality", "quality_label"], axis=1)
y=df["quality_label"]

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2
)

clf=RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))