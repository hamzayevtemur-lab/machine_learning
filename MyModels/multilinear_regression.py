import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class MyLinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
        
    def fit(self, X, y):
        n_samples, n_features=X.shape
        
        #initialize
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iters):
            y_pred=np.dot(X, self.weights)+self.bias
            
            #gradients
            dw=(1/n_samples)*np.dot(X.T, (y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)
            
            #update
            self.weights-=self.lr*dw
            self.bias-=self.lr*db
            
    def predict(self, X):
        return np.dot(X, self.weights)+self.bias #type:ignore
    
    def score(self, X, y):
        y_pred=self.predict(X)
        
        ss_total=np.sum((y-np.mean(y))**2)
        ss_residual=np.sum((y-y_pred)**2)
        
        return 1-(ss_residual/ss_total)
    


#Load dataset
df=pd.read_csv("/Users/mac/Desktop/Machine Learning/datasets/car_price.csv")

print("First 5 rows:")
print(df.head())

print("Dataset information:")
print(df.info())

print("Dataset summary:")
print(df.describe())

print("\nColumns with unusual distribution or outliers:")
for col in df.columns:
    print(f"Number of unique values in {col}: {df[col].nunique()}")

#Drop useless Columns
df=df.drop(["car_ID", "CarName"], axis=1)


# 2. BINARY ENCODING (0 / 1)
binary_cols=["fueltype", "aspiration", "doornumber", "enginelocation"]

for col in binary_cols:
    df[col]=df[col].astype("category").cat.codes
    
# 3. ONE-HOT ENCODING (MULTI-CATEGORY)
multi_cols=['carbody', 'drivewheel', 'enginetype',
              'cylindernumber', 'fuelsystem']

df=pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
 
# Data evaluation       
X=df.drop("price", axis=1)
y=df["price"]


#Train test split
X_train, X_test, y_train,y_test=train_test_split(
    X,y,test_size=0.2, random_state=42)



#Feature Scaling
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)

#Train my model
model=MyLinearRegression(lr=0.001, n_iters=15000)

model.fit(X_train_scaled, y_train)

preds=model.predict(X_test_scaled)

print("Predictions:", preds)
print("R2 Score:", model.score(X_test_scaled, y_test))  

# Train Sklearn model  
sk_model=LinearRegression()
sk_model.fit(X_train_scaled, y_train)
sk_preds=sk_model.predict(X_test_scaled)

print("SKlearn Predictions:", sk_preds)
print("Sklearn R2 Score:", r2_score(y_test, sk_preds))  
