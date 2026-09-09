import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


digits=load_digits()
X=digits.data    #type: ignore
y=digits.target  #type: ignore

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# Show sample images
fig, axes=plt.subplots(2,5, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray") #type: ignore
    ax.set_title(f"Label:{y[i]}")
    ax.axis("off")
    
plt.show()


# Data Exploration

print("Number of samples:", len(X))
print("Number of features:", X.shape[1])

#Class distribution
unique, counts=np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))



# Train-test Split + Scaling

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# Train SVM (Linear Kernel)

svm_linear=SVC(kernel="linear")
svm_linear.fit(X_train_scaled, y_train)

y_pred=svm_linear.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Confusion Matrix

cm=confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Linear SVM):")
plt.show()


#Polynomial
svm_poly=SVC(kernel="poly", degree=3)
svm_poly.fit(X_train_scaled, y_train)
poly_pred=svm_poly.predict(X_test_scaled)

#RBF
svm_rbf=SVC(kernel="rbf")
svm_rbf.fit(X_train_scaled, y_train)
rbf_pred=svm_rbf.predict(X_test_scaled)

print("Poly Accuracy:", accuracy_score(y_test, poly_pred))
print("RBF Accuracy:", accuracy_score(y_test, rbf_pred))


# Hyperparameter Tuning (GridSearchCV)

param_grid={
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf', 'poly'],
    'degree': [2, 3]  # only used for poly
}

grid=GridSearchCV(SVC(), param_grid, cv=3, verbose=2, n_jobs=2)
grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_
best_pred = best_model.predict(X_test_scaled)

print("Best Accuracy:", accuracy_score(y_test, best_pred))


# PCA Visualization (2D)

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled:=scaler.fit_transform(X))

plt.figure(figsize=(8, 6))
scatter=plt.scatter(X_pca[:,0], X_pca[:, 1], c=y, cmap="tab10")
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title("PCA Visualization of Digits")
plt.show()


# Compare with Logistic Regression

log_reg=LogisticRegression(max_iter=5000)
log_reg.fit(X_train_scaled, y_train)

log_pred=log_reg.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))

