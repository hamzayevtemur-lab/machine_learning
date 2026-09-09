from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Ensemble
from sklearn.ensemble import VotingClassifier

# Load dataset
data=load_wine()
X, y=data.data, data.target #type:ignore

# Train-test split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)


########## Baseline Models

# Initialize models
log_model=LogisticRegression(max_iter=5000)
knn_model=KNeighborsClassifier(n_neighbors=5)
dt_model=DecisionTreeClassifier(random_state=42)

# Train models
log_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Predictions
log_pred=log_model.predict(X_test)
knn_pred=knn_model.predict(X_test)
dt_pred=dt_model.predict(X_test)

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))




###### Hard Voting Classifier
hard_voting=VotingClassifier(
    estimators=[
        ("lr", log_model),
        ("knn", knn_model),
        ("dt", dt_model)
    ],
    voting="hard"
)

# Train 
hard_voting.fit(X_train, y_train)

#Predict
hard_pred=hard_voting.predict(X_test)

# Accuracy
print("Hard Voting Accuracy:", accuracy_score(y_test, hard_pred))



###### Soft Voting

soft_voting=VotingClassifier(
    estimators=[
        ("lr", log_model),
        ("knn", knn_model),
        ("dt", dt_model)
    ],
    voting="soft"
)

# Train 
soft_voting.fit(X_train, y_train)

# Predict
soft_pred=soft_voting.predict(X_test)

# Accuracy
print("Soft Voting Accuracy:", accuracy_score(y_test, soft_pred))

