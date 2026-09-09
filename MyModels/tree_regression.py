import numpy as np

class MyDecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.tree=None
        
    # Mean Squared Erro
    def mse(self, y):
        return np.mean((y-np.mean(y))**2)
    
    # Best Split
    def best_split(self, X, y):
        best_feature, best_threshold=None, None
        best_mse=float("inf")
        
        n_samples, n_features=X.shape
        
        for feature in range(n_features):
            thresholds=np.unique(X[:, feature])
            
            for t in thresholds:
                left_idx=X[:, feature]<=t
                right_idx=X[:,feature]>t
                
                if sum(left_idx)==0 or sum(right_idx)==0:
                    continue
                
                mse_left=self.mse(y[left_idx])
                mse_right=self.mse(y[right_idx])
                
                weighted_mse=(
                    len(y[left_idx])/n_samples*mse_left+
                    len(y[right_idx])/n_samples*mse_right
                )
                
                if weighted_mse<best_mse:
                    best_mse=weighted_mse
                    best_feature=feature
                    best_threshold=t
                    
        return best_feature, best_threshold
    
    
    # Build Tree
    def build_tree(self, X, y, depth=0):
        if (len(y)<self.min_samples_split or depth>=self.max_depth):
            return np.mean(y)
        
        feature, threshold=self.best_split(X, y)
        
        if feature is None:
            return np.mean(y)
        
        left_idx=X[:, feature]<=threshold
        right_idx=X[:, feature]>threshold
        
        left_subtree=self.build_tree(X[left_idx], y[left_idx], depth+1)
        right_subtree=self.build_tree(X[right_idx], y[right_idx], depth+1)
        
        return {
            "feature":feature,
            "threshold": threshold,
            "left": left_subtree,
            "right":right_subtree
        }
        
    
    # Fit
    def fit(self, X, y):
        self.tree=self.build_tree(X,y)
      
      
    # Predict  
    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        
        if x[node["feature"]]<=node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])
        
    def predict(self, X):
        return np.array([self.predict_one(x , self.tree) for x in X])
    
    
    
    
########################## My Model #######################

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

data = load_diabetes()
X, y = data.data, data.target #type:ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

my_model=MyDecisionTreeRegressor(max_depth=6)
my_model.fit(X_train, y_train)

my_pred=my_model.predict(X_test)




############################ SKlearn Model ###############
from sklearn.tree import DecisionTreeRegressor

sk_model=DecisionTreeRegressor(max_depth=6, random_state=42)
sk_model.fit(X_train, y_train)

sk_pred=sk_model.predict(X_test)


############## GridSearch ##########
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [2, 4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)


################### Evaluate ##############

from sklearn.metrics import r2_score, mean_squared_error

print("My Model R2:", r2_score(y_test, my_pred))
print("Sklearn R2:", r2_score(y_test, sk_pred))

print("\nMy Model MSE:", mean_squared_error(y_test, my_pred))
print("Sklearn MSE:", mean_squared_error(y_test, sk_pred))


best_model = grid.best_estimator_

preds = best_model.predict(X_test)


print("Improved R2:", r2_score(y_test, preds))
print("Improved MSE:", mean_squared_error(y_test, preds))




