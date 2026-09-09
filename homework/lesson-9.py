import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load Dataset
data=load_breast_cancer()

#Covert to DataFrame
df=pd.DataFrame(data.data, columns=data.feature_names) #type:ignore
df["target"]=data.target #type:ignore

#Display Info
print(df.head())
print("Shape:", df.shape)
print("Features:", data.feature_names) #type:ignore
print("Target distribution:\n", df["target"].value_counts())

# Train-Test Split & Scaling
X=df.drop("target", axis=1)
y=df["target"]

#Split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Model Training
lr=LogisticRegression()
rf=RandomForestClassifier()

#Train 
lr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train) #Rf doest need scaling

#Predictions
y_pred_lr=lr.predict(X_test_scaled)
y_pred_rf=rf.predict(X_test)


# Metrics Computation

#Logistic Regression Metrics
metrics_lr={
    "Accuracy": accuracy_score(y_test, y_pred_lr),
    "Precision": precision_score(y_test, y_pred_lr),
    "Recall": recall_score(y_test, y_pred_lr),
    "F1": f1_score(y_test, y_pred_lr)
}

# Random Forest Metrics
metrics_rf = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1": f1_score(y_test, y_pred_rf)
}

print("Logistic Regression:", metrics_lr)
print("Random Forest:", metrics_rf)

# Confusion Matrix
cm_lr=confusion_matrix(y_test, y_pred_lr)
cm_rf=confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")

plt.subplot(1,2,2)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")

plt.show()


# Classification Report
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# ROC Curve & AUC
y_prob_lr=lr.predict_proba(X_test_scaled)[:,1]
y_prob_rf=rf.predict_proba(X_test)[:,1]

#ROC
fpr_lr, tpr_lr, _ =roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ =roc_curve(y_test, y_prob_rf)

#AUC
auc_lr=auc(fpr_lr, tpr_lr)
auc_rf=auc(fpr_rf, tpr_rf)

#PLot
plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={auc_lr:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC={auc_rf:.2f})")
plt.plot([0,1],[0,1],'--')  # random line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Bar Chart of Metrics
labels=list(metrics_lr.keys())

lr_values = list(metrics_lr.values())
rf_values = list(metrics_rf.values())

x = np.arange(len(labels))

plt.figure()
plt.bar(x - 0.2, lr_values, width=0.4, label="LogReg")
plt.bar(x + 0.2, rf_values, width=0.4, label="RandomForest")

plt.xticks(x, labels)
plt.legend()
plt.title("Model Comparison")
plt.show()

# Which model performed better overall? In my case Logistic Regression
# Which model had better Recall? Both models are equally strong for cancer detection 
# Which model had higher Precision? Logistic Regression is more reliable when predicting “benign”
# Which model had better AUC? Logistic Regression also has slightly better AUC
# Does accuracy tell the full story? No, accuracy is NOT enough.


# Bonus (Threshold Tuning)
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

precision_list = []
recall_list = []
f1_list = []

for t in thresholds:
    y_pred_thresh=(y_prob_lr >= t).astype(int)
    
    precision_list.append(precision_score(y_test, y_pred_thresh))
    recall_list.append(recall_score(y_test, y_pred_thresh))
    f1_list.append(f1_score(y_test, y_pred_thresh))

# Plot
plt.plot(thresholds, precision_list, label="Precision")
plt.plot(thresholds, recall_list, label="Recall")
plt.plot(thresholds, f1_list, label="F1")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Threshold Tuning")
plt.show()

