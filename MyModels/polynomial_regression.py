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


######################## Poly Nomial ######################

import numpy as np
from itertools import combinations_with_replacement

class Poly:
    def __init__(self, degree):
        self.degree=degree
        
    def fit(self, X):
        self.n_features=X.shape[1]
        self.combinations=[]
        
        for d in range(1, self.degree+1):
            self.combinations+=list(
                combinations_with_replacement(range(self.n_features), d)
            )
            
    def transform(self, X):
        X_poly=[]
        
        for x in X:
            features=[]
            for comb in self.combinations:
                val=1
                for i in comb:
                    val*=x[i]
                features.append(val)
                
            X_poly.append(features)
            
        return np.array(X_poly) 
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class MyPoly:
    def __init__(self, degree=2, lr=0.01, n_iters=1000):
        self.degree=degree
        self.lr=lr
        self.n_iters=n_iters
        
        self.poly=Poly(degree)
        self.model=MyLinearRegression(lr=lr, n_iters=n_iters)
        
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def score(self, X, y):
        X_poly = self.poly.transform(X)
        return self.model.score(X_poly, y)
    
    
poly_model=MyPoly(degree=2, lr=0.0001, n_iters=10000)
poly_model.fit(X_train_scaled, y_train)

poly_pred=poly_model.predict(X_test_scaled)


print("Polynomial R:", r2_score(y_test, poly_pred))


########### Sklearn Polynomial regression #############
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2 ,include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 2: scaling
scaler_for_poly = StandardScaler()
X_train_poly = scaler_for_poly.fit_transform(X_train_poly)
X_test_poly = scaler_for_poly.transform(X_test_poly)

sk_poly_model = LinearRegression()
sk_poly_model.fit(X_train_poly, y_train)

sk_poly_preds = sk_poly_model.predict(X_test_poly)

print("Sklearn Polynomial R:", r2_score(y_test, sk_poly_preds))



print("\n===== MODEL COMPARISON =====")

print("My Linear R:", model.score(X_test_scaled, y_test))
print("Sklearn Linear R:", r2_score(y_test, sk_preds))

print("My Polynomial R:", r2_score(y_test, poly_pred))
print("Sklearn Polynomial R:", r2_score(y_test, sk_poly_preds))
        
    