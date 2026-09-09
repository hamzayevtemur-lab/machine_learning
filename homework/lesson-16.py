# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline



# 2. LOAD DATA
df = pd.read_csv("/Users/mac/Desktop/Machine Learning/ai-roadmap/content/ml/supervised/11 k nearest neighbour/data/TelcoCustomerChurn.csv")

print("First 5 rows:\n", df.head())
print("\nInfo:\n")
df.info()



# 3. DATA EXPLORATION
print("\nTarget distribution:\n", df["Churn"].value_counts())

# Correlation (numerical only)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Group analysis
print("\nChurn by Contract:\n")
print(df.groupby("Contract")["Churn"].value_counts(normalize=True))



# 4. DATA CLEANING

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop unnecessary column safely
df.drop("customerID", axis=1, inplace=True, errors="ignore")

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)


# 5. SPLIT FEATURES & TARGET
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]


# 6. TRAIN-TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# 7. SCALING 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 8. TRAIN MODEL
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# 9. EVALUATION
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 10. OPTIMIZE K
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot K vs Accuracy
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.show()


# 11. CROSS-VALIDATION 
scores = []

for k in k_values:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    
    cv_score = cross_val_score(pipeline, X, y, cv=5)
    scores.append(cv_score.mean())

optimal_k = k_values[np.argmax(scores)]
print("\nBest K from Cross-Validation:", optimal_k)


# 12. DISTANCE METRIC COMPARISON
metrics = ["euclidean", "manhattan", "minkowski"]

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"{metric}: {accuracy_score(y_test, y_pred)}")
    
    
    

# KNN pipeline
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=optimal_k))
])

# Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

knn_scores = cross_val_score(knn_pipeline, X, y, cv=5)
lr_scores = cross_val_score(lr_pipeline, X, y, cv=5)

print("KNN CV Accuracy:", knn_scores.mean())
print("Logistic Regression CV Accuracy:", lr_scores.mean())
