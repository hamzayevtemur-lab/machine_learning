import numpy as np
from collections import Counter

class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.tree=None
        
    
    # Gini Impurity    
    def gini(self, y):
        counts=Counter(y)
        impurity=1
        for c in counts.values():
            p=c/len(y)
            impurity-=p**2
        return impurity
    
    # Best Split
    def best_split(self, X, y):
        best_feature, best_threshold=None, None
        best_gini=float("inf")
        
        n_samples, n_features=X.shape
        
        for feature in range(n_features):
            thresholds=np.unique(X[:, feature])
            
            for t in thresholds:
                left_idx=X[:, feature]<=t
                right_idx=X[:,feature]>t
                
                if sum(left_idx)==0 or sum(right_idx)==0:
                    continue
                
                gini_left=self.gini(y[left_idx])
                gini_right=self.gini(y[right_idx])
                
                weighted_gini=(
                    len(y[left_idx])/n_samples*gini_left+
                    len(y[right_idx])/n_samples*gini_right
                )
                
                if weighted_gini<best_gini:
                    best_gini=weighted_gini
                    best_feature=feature
                    best_threshold=t
        
        return best_feature, best_threshold
    
    # Build Tree
    def build_tree(self, X, y, depth=0):
        if (len(set(y))==1 or len(y)<self.min_samples_split or 
            depth>=self.max_depth):
            
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold=self.best_split(X, y)
        
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        left_idx=X[:, feature]<=threshold
        right_idx=X[:, feature]>threshold
        
        left_subtree=self.build_tree(X[left_idx], y[left_idx], depth+1)
        right_subtree=self.build_tree(X[right_idx], y[right_idx], depth+1)
        
        return {
            "feature":feature, 
            "threshold":threshold,
            "left":left_subtree,
            "right":right_subtree
        }
    
    # Fit
    def fit(self, X, y):
        self.tree=self.build_tree(X, y)
        
    
    # Predict one
    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        
        if x[node["feature"]]<=node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])
        
    # Predict
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])
    
    
    
    
    

################# My Model ##########################  
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=load_iris()
X, y=data.data, data.target #type:ignore

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

my_model=DecisionTreeClassifierScratch(max_depth=3)
my_model.fit(X_train, y_train)

my_pred=my_model.predict(X_test)

print("My Model Accuracy:", accuracy_score(y_test, my_pred))






################### Sklearn Model ########################
from sklearn.tree import DecisionTreeClassifier

sk_model=DecisionTreeClassifier(
    max_depth=3,
    criterion="gini",
    random_state=42
)

sk_model.fit(X_train, y_train)

sk_pred=sk_model.predict(X_test)

print("Sklearn Accuracy:", accuracy_score(y_test, sk_pred))

# Visualize sklearn tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plot_tree(sk_model, filled=True)
plt.show()