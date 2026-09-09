import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV, 
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import warnings 
warnings.filterwarnings("ignore")


# Load dataset from OpenML
cifar = fetch_openml('CIFAR_10_small')

# Features and labels
X = cifar.data
y = cifar.target

print("Dataset Shape:", X.shape)
print("Labels Shape:", y.shape)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

## Display Sample Images
plt.figure(figsize=(10, 5))

for i in range(10):
    image = X_train.iloc[i].values.reshape(32, 32, 3)
    plt.subplot(2, 5, i + 1)
    plt.imshow(image.astype(np.uint8), cmap="gray")
    plt.title(y_train.iloc[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# Class Names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# Feature Scaling
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# PCA
pca=PCA(n_components=200)
X_train_pca=pca.fit_transform(X_train_scaled)
X_test_pca=pca.transform(X_test_scaled)

print(X_train_pca.shape)
print(X_test_pca.shape)

# Explained Variance
explained_variance=np.sum(pca.explained_variance_ratio_)

print("Explained Variance:", explained_variance)

# PCA Variance Graph
cumulative_variance=np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 5))

plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.grid(True)
 
plt.show()



###################### Logistic Regression (Baseline Model)
log_model=Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=200)),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))  
])


log_model.fit(X_train, y_train)

log_pred=log_model.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Logistic Regression Precision:", precision_score(y_test, log_pred, average='weighted'))
print("Logistic Regression Recall:", recall_score(y_test, log_pred, average='weighted'))
print("Logistic Regression F1 Score:", f1_score(y_test, log_pred, average='weighted'))



#################### Random Forest
rf_model=Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=200)),
    ("model", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

rf_pred=rf_model.predict(X_test)

#Evaluate Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Precision:", precision_score(y_test, rf_pred, average='weighted'))
print("Random Forest Recall:", recall_score(y_test, rf_pred, average='weighted'))
print("Random Forest F1 Score:", f1_score(y_test, rf_pred, average='weighted'))




#################### Extra Trees
extra_model=Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=200)),
    ("model", ExtraTreesClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ))  
])

extra_model.fit(X_train, y_train)

extra_pred=extra_model.predict(X_test)

# Evaluate Extra Trees
print("Extra Trees Accuracy:", accuracy_score(y_test, extra_pred))
print("Extra Trees Precision:", precision_score(y_test, extra_pred, average='weighted'))
print("Extra Trees Recall:", recall_score(y_test, extra_pred, average='weighted'))
print("Extra Trees F1 Score:", f1_score(y_test, extra_pred, average='weighted'))



################## Gradient Boosting
gb_model=Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=200)),
    ("model", GradientBoostingClassifier(
        n_estimators=20,
        learning_rate=0.1,
        random_state=42
    ))
]) 

gb_model.fit(X_train, y_train)

gb_pred=gb_model.predict(X_test)

# Evaluate Gradient Boosting
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Gradient Boosting Precision:", precision_score(y_test, gb_pred, average='weighted'))
print("Gradient Boosting Recall:", recall_score(y_test, gb_pred, average='weighted'))
print("Gradient Boosting F1 Score:", f1_score(y_test, gb_pred, average='weighted'))


################## Bagging
bag_model=Pipeline([
    ("scaler", StandardScaler()),   
    ("pca", PCA(n_components=200)),
    ("model", BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42,
        n_jobs=-1  
    ))
])  

bag_model.fit(X_train, y_train)
bag_pred=bag_model.predict(X_test)

# Evaluate Bagging
print("Bagging Accuracy:", accuracy_score(y_test, bag_pred))
print("Bagging Precision:", precision_score(y_test, bag_pred, average='weighted'))
print("Bagging Recall:", recall_score(y_test, bag_pred, average='weighted'))
print("Bagging F1 Score:", f1_score(y_test, bag_pred, average='weighted'))


################### Voting Classifier
voting_model=Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=200)), 
    ("model", VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("et", ExtraTreesClassifier(n_estimators=100))
        ],
        voting="hard",
        n_jobs=-1
    ))
])   
                       

voting_model.fit(X_train, y_train)
vote_pred=voting_model.predict(X_test)

# Evaluate Voting Classifier
print("Voting Accuracy:", accuracy_score(y_test, vote_pred))
print("Voting Precision:", precision_score(y_test, vote_pred, average='weighted'))
print("Voting Recall:", recall_score(y_test, vote_pred, average='weighted'))
print("Voting F1 Score:", f1_score(y_test, vote_pred, average='weighted'))




#################### KNN
knn_model=Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=200)),
    ("model", KNeighborsClassifier(n_neighbors=5))
]) 
knn_model.fit(X_train, y_train)

knn_pred=knn_model.predict(X_test)

# Evaluate KNN
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Precision:", precision_score(y_test, knn_pred, average='weighted'))
print("KNN Recall:", recall_score(y_test, knn_pred, average='weighted'))
print("KNN F1 Score:", f1_score(y_test, knn_pred, average='weighted'))



################ GridSearchCV
param_grid={
    "n_estimators":[50, 100],
    "max_depth": [None, 20],
    "min_samples_split":[2, 5]
}

grid_model=GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_model.fit(X_train_pca, y_train)

print("Best Parameters:", grid_model.best_params_)
print("Best Score:", grid_model.best_score_)



############ SVC
svm_pipeline=Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=200)),
    ("model", SVC(
        kernel="rbf",
        C=5,
        gamma="scale"
    ))
])

svm_pipeline.fit(X_train, y_train)

svm_pred=svm_pipeline.predict(X_test)

# Evaluate SVC
print("SVC Accuracy:", accuracy_score(y_test, svm_pred))
print("SVC Precision:", precision_score(y_test, svm_pred, average='weighted'))
print("SVC Recall:", recall_score(y_test, svm_pred, average='weighted'))
print("SVC F1 Score:", f1_score(y_test, svm_pred, average='weighted'))  



############## Stack Model
stack_model=StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=100,
            random_state=42
        )),
        ("knn", KNeighborsClassifier(
            n_neighbors=5
        ))
    ],
    
    final_estimator=LogisticRegression(),
    n_jobs=-1
)

# Stacking Pipeline
stack_pipeline=Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=200)),
    ("model", stack_model)
])

stack_pipeline.fit(X_train, y_train)

stack_pred=stack_pipeline.predict(X_test)

# Evaluate Stack Model
print("Stack Accuracy:", accuracy_score(y_test, stack_pred))
print("Stack Precision:", precision_score(y_test, stack_pred, average='weighted'))
print("Stack Recall:", recall_score(y_test, stack_pred, average='weighted'))
print("Stack F1 Score:", f1_score(y_test, stack_pred, average='weighted'))  


################ Cross Validation
cv_scores=cross_val_score(
    rf_model,
    X_train_pca,
    y_train,
    cv=5,
    scoring="accuracy"
)
print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())



#### Confusion Matrix
def plot_confusion_matrix(model_name, y_test, predictions):
    cm=confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 8))
    
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    ).plot(cmap="Blues", xticks_rotation=45)
    
    plt.title(f"{model_name} Confusion Matrix")
    
    plt.show()


## Classification Report Function

def show_classification_report(model_name, y_test, predictions):
    print(f"\n{model_name} Classification Report")
    
    print(classification_report(
        y_test,
        predictions
    ))
    
models_predictions = {
    "Logistic Regression": log_pred,
    "Random Forest": rf_pred,
    "Extra Trees": extra_pred,
    "Gradient Boosting": gb_pred,
    "Bagging": bag_pred,
    "Voting Classifier": vote_pred,
    "KNN": knn_pred,
    "SVC": svm_pred,
    "Stack": stack_pred
}

for model_name, predictions in models_predictions.items():
    # Confusion Matrix
    plot_confusion_matrix(
        model_name, 
        y_test,
        predictions)
    
    # Classification Report
    show_classification_report(
        model_name,
        y_test,
        predictions
    )


# Compare Models
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "Bagging",
        "Voting",
        "KNN",
        "SVC",
        "Stack"
    ],

    "Accuracy": [
        accuracy_score(y_test, log_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, extra_pred),
        accuracy_score(y_test, gb_pred),
        accuracy_score(y_test, bag_pred),
        accuracy_score(y_test, vote_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, stack_pred)
    ]
})

print(results)

# Accuracy Comparison Graph
plt.figure(figsize=(12, 6))

plt.bar(results["Model"],
        results["Accuracy"])

plt.xticks(rotation=45)

plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.show()