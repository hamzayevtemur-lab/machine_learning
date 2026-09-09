from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

mnist=fetch_openml("mnist_784", version=1)

X=mnist.data 
y=mnist.target.astype(int)

# Normalize 
X=X/255.0

# Train / Test Split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression (Baseline)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_model=LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log=log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))


# KNN 
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred_knn=knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))



# SVM 
from sklearn.svm import SVC

svm=SVC(kernel="rbf")
svm.fit(X_train, y_train)

y_pred_svm=svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))



# Decision Tree 
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(max_depth=20, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt=dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))


# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(
    n_estimators=100, 
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf=rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


######### Confusion Matrix ############

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

models={
    "Logistic":log_model,
    "KNN":knn,
    "SVM":svm,
    "Decision Tree":dt,
    "Random Forest":rf
}

for name, model in models.items():
    y_pred=model.predict(X_test)
    cm=confusion_matrix(y_test, y_pred)
    
    disp=ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(name)
    plt.show()
    
    

## Comparison Table
results={}

for name, model in models.items():
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    results[name]=acc

for name, acc in results.items():
    print(f"{name}:{acc:.4f}")
    
# Bar Chart
plt.bar(results.keys(), results.values()) #type:ignore
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.title("Model Comparison on MNIST")
plt.show()

