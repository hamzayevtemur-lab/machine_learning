import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Load dataset
iris=load_iris()
X=iris.data #type:ignore
y=iris.target #type:ignore

feature_names=iris.feature_names #type:ignore
target_names=iris.target_names #type:ignore

# Convert to DataFrame
df=pd.DataFrame(X, columns=feature_names)
df["target"]=y

print("First 5 rows:", df.head())
print("Info:", df.info())
print("Summary:", df.describe())

# Visualization
sns.pairplot(df, hue="target")
plt.show()


# Data Preparation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Building
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred=model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Tree Visualization
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(
    model, 
    feature_names=feature_names,
    class_names=target_names,
    filled=True
)
plt.show()

# Feature Importance
importances=model.feature_importances_

for name, importance in zip(feature_names, importances):
    print(f"{name}:{importance:.4f}")
    
# Visualization
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid={
    "max_depth":[2, 3,4,5],
    "criterion":["gini", "entropy"],
    "min_samples_split":[2,5,10]
}

grid=GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5
    )

grid.fit(X_train, y_train)

print("Best param:", grid.best_params_)



# Decision Boundary Visualization (2 features only)

from matplotlib.colors import ListedColormap

X_small=X[:, [2,3]] #petal length & width

X_train_s, X_test_s, y_train_s, y_test_s=train_test_split(
    X_small, y, test_size=0.2, random_state=42
)
model_small=DecisionTreeClassifier(max_depth=3)
model_small.fit(X_train_s, y_train_s)

# Meshgrid
x_min, x_max=X_small[:,0].min()-1, X_small[:,0].max()+1
y_min, y_max=X_small[:, 1].min()-1, X_small[:, 1].max()+1

xx, yy=np.meshgrid(np.arange(x_min, x_max, 0.02),
                   np.arange(y_min, y_max, 0.02))

Z=model_small.predict(np.c_[xx.ravel(), yy.ravel()])
Z=Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_small[:, 0], X_small[:, 1], c=y)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Decision Boundaries")
plt.show()