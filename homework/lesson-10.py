import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_digits


#PART 1 — Data Loading & Exploration

#Load dataset
digits=load_digits()

X=digits.data #type:ignore
y=digits.target #type:ignore



# Basic info
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Unique classes:", np.unique(y))
print("Number of classes:", len(np.unique(y)))

# Pixel stats
print("Min pixel:", np.min(X))
print("Max pixel:", np.max(X))

# SHow one Image
plt.matshow(digits.images[0]) #type:ignore
plt.title(f"Label:{y[0]}")
plt.show()


#PART 2 — Train/Test Split + Scaling

#Split
X_train, X_test, y_train, y_test=train_test_split(
    X,y, test_size=0.2, random_state=42
)

#Scale
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# PART 3 — Train Models

# Logistic Regression (OvR)
log_reg=LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)


# SVM (OvO)
svm=SVC(kernel="rbf", probability=True)
svm.fit(X_train_scaled, y_train)

# Random Forest
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


# PART 4 — Predictions & Evaluation

#Predictions
y_pred_log=log_reg.predict(X_test_scaled)
y_pred_svm=svm.predict(X_test_scaled)

#Accuracy
print("Logistic Accuracy:", accuracy_score(y_test, y_pred_log))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


#Confusion Matrix
cm_log=confusion_matrix(y_test, y_pred_log)
cm_svm=confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(8,6))
sns.heatmap(cm_log, annot=True, fmt="d")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(cm_svm, annot=True, fmt="d")
plt.title("SVM Confusion Matrix")
plt.show()

# Classification Report
print("Logistic Regression:\n", classification_report(y_test, y_pred_log))
print("SVM:\n", classification_report(y_test, y_pred_svm))


# PART 5 — Advanced Metrics

#Precision

# Logistic
print("Logistic Macro Precision:", precision_score(y_test, y_pred_log, average='macro'))
print("Logistic Weighted Precision:", precision_score(y_test, y_pred_log, average='weighted'))

# SVM
print("SVM Macro Precision:", precision_score(y_test, y_pred_svm, average='macro'))
print("SVM Weighted Precision:", precision_score(y_test, y_pred_svm, average='weighted'))


#f1-score

# Logistic
print("Logistic Macro f1-score:", f1_score(y_test, y_pred_log, average='macro'))
print("Logistic Weighted f1-score:", f1_score(y_test, y_pred_log, average='weighted'))

# SVM
print("SVM Macro f1-score:", f1_score(y_test, y_pred_svm, average='macro'))
print("SVM Weighted Precision:", f1_score(y_test, y_pred_svm, average='weighted'))

#Recall

# Logistic
print("Logistic Macro Recall", recall_score(y_test, y_pred_log, average='macro'))
print("Logistic Weighted Recall:", recall_score(y_test, y_pred_log, average='weighted'))

# SVM
print("SVM Macro Recall:", recall_score(y_test, y_pred_svm, average='macro'))
print("SVM Weighted Recall:", recall_score(y_test, y_pred_svm, average='weighted'))


# Confusion Matrix Heatmap
def plot_confusion_matrix(y_true, y_pred, title):
    cm=confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    

# Logistic Regresssion
plot_confusion_matrix(y_test, y_pred_log, "Logistic Regression Confusion Matrix")

#SVM
plot_confusion_matrix(y_test, y_pred_svm, "SVM Confusion Matrix")

### ROC Curve (Multiclass - One-vs-Rest)
def plot_multiclass_roc(model, X_test, y_test, title):
    # Binarize labels
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    # Get probabilities
    y_score = model.predict_proba(X_test)

    plt.figure(figsize=(10,8))

    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i]) #type:ignore
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    # Diagonal line
    plt.plot([0,1], [0,1], 'k--')

    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()
    

# Logistic Regression ROC
plot_multiclass_roc(log_reg, X_test_scaled, y_test, "ROC Curve - Logistic Regression")

# SVM ROC (make sure probability=True)
plot_multiclass_roc(svm, X_test_scaled, y_test, "ROC Curve - SVM")


# Sample Predictions Visualization (10 Images)

def  plot_sample_predictions(X_test, y_test, y_pred, n=10):
    indices=random.sample(range(len(X_test)), n)
    
    plt.figure(figsize=(12,6))

    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)

        # Reshape image (64 → 8x8)
        image = X_test[idx].reshape(8, 8)

        pred = y_pred[idx]
        true = y_test[idx]

        # Color for correctness
        color = 'green' if pred == true else 'red'

        plt.imshow(image, cmap='gray')
        plt.title(f"P:{pred} / T:{true}", color=color)

        # Border color
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        plt.axis('off')

    plt.suptitle("Sample Predictions (Green = Correct, Red = Wrong)")
    plt.show()


# Logistic Regression samples
plot_sample_predictions(X_test, y_test, y_pred_log)

# SVM samples
plot_sample_predictions(X_test, y_test, y_pred_svm)
    
    