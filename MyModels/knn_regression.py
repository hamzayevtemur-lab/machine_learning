import numpy as np

class KNNRegressor:
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
        # Compute distance
        distances=[self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices=np.argsort(distances)[: self.k]
        k_nearest_values=[self.y_train[i] for i in k_indices]
        
        # Average for regression
        return np.mean(k_nearest_values)
    
    def score(self, y_true, y_pred):
        # convert to numpy arrays
        y_true=np.array(y_true)
        y_pred=np.array(y_pred)
        
        #Mean of actual values
        y_mean=np.mean(y_true)
        
        # Total sum of squares
        ss_total=np.sum((y_true-y_mean)**2)
        
        # Residual Sum of Squares
        ss_residual=np.sum((y_true-y_pred)**2)
        
        #r2 calculation
        r2=1-(ss_residual/ss_total)
        
        return r2
        

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score

X, y = make_regression( #type:ignore
    n_samples=500,
    n_features=5,
    noise=20,
    random_state=42
) 

# Split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

#scale
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Train custom KNN
knn_reg=KNNRegressor(k=5)
knn_reg.fit(X_train, y_train)

#Predict
y_pred=knn_reg.predict(X_test)

#MSE
mse=np.mean((y_test-y_pred)**2)
print("MSE:", mse)

#R2 score
print("Score:", knn_reg.score(y_test, y_pred))

######### Sklearn Model ##########
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=5))
    ]),
    
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ]),
    
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}


results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results[name] = (r2, mse, rmse)
    
    print(f"\n=== {name} ===")
    print("R²:", r2)
    print("MSE:", mse)
    print("RMSE:", rmse)
   
   
    
print("\n=== Cross Validation (R²) ===")

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name}: {scores.mean()}")