
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k=k
        
    def fit(self, X, y):
        self.X_train=X
        self.y_train=y
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self, X):
        predictions=[self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # compute distance
        distance = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get k nearest neighbors
        k_indices=np.argsort(distance)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
data=load_iris()
X, y=data.data, data.target #type:ignore

#Split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Train custom KNN
knn=KNNClassifier(k=5)
knn.fit(X_train, y_train)

#Predict
y_pred=knn.predict(X_test)

# Accuracy
accuracy=np.mean(y_pred==y_test)
print("Accuracy:", accuracy)
print("Accuracy with accuracy score function:", accuracy_score(y_test, y_pred))

    
############ SKLearn Models ###############
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)    


y_pred_sk = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_sk))