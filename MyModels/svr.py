import numpy as np
import pandas as pd

class MySVR:
    def __init__(self, lr=0.001, lambda_param=0.01, epsilon=0.1, n_iters=1000):
        self.lr=lr
        self.lambda_param=lambda_param
        self.epsilon=epsilon
        self.n_iters=n_iters
        self.w=None
        self.b=None
        
    def fit(self, X, y):
        n_samples, n_features=X.shape
        
        self.w=np.zeros(n_features)
        self.b=0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_pred=np.dot(x_i, self.w)+self.b
                error=y_pred-y[idx]
                
                if abs(error)<=self.epsilon:
                    self.w-=self.lr*(2*self.lambda_param*self.w)
                    
                else:
                    grad = np.sign(error)
                    
                    self.w -= self.lr * (2*self.lambda_param*self.w + error * x_i)
                    self.b -= self.lr * error
                    
    def predict(self,X):
        return np.dot(X, self.w)+self.b #type:ignore
    

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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


# 4. CHECK RESULT
print(df.head())
print("\nAll columns are numeric now:", df.dtypes.unique())
print("all columns:", df.columns)
print(df.shape)

X=df.drop("price", axis=1).values
y=df["price"].values


#Train test split
X_train, X_test, y_train,y_test=train_test_split(
    X,y,test_size=0.2, random_state=42)


# Scale
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
y_test = y_scaler.transform(y_test.reshape(-1,1)).flatten()



###################### My Model ###########
# Train SVR
MyModel=MySVR(lr=0.01, lambda_param=0.01, epsilon=0.1, n_iters=1000)
MyModel.fit(X_train, y_train)

# Predict
y_pred_my=MyModel.predict(X_test)
y_pred_my = y_scaler.inverse_transform(y_pred_my.reshape(-1,1)).flatten()


############## SKlearn SVR Model ###########
from sklearn.svm import SVR

sk_model = SVR(
    kernel="rbf",
    C=10,
    gamma="scale",
    epsilon=0.1
)

sk_model.fit(X_train, y_train)

y_pred_sk=sk_model.predict(X_test)
y_pred_sk = y_scaler.inverse_transform(y_pred_sk.reshape(-1,1)).flatten()



############## SKlearn linearregression Model ###########
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr=lr.predict(X_test)
y_pred_lr = y_scaler.inverse_transform(y_pred_lr.reshape(-1,1)).flatten()


y_test = y_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

######### Compare Models ######
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("R2:", r2_score(y_true, y_pred))
    print()

evaluate("My SVR", y_test, y_pred_my)
evaluate("Sklearn SVR", y_test, y_pred_sk)
evaluate("Linear Regression", y_test, y_pred_lr)


###### Visualization 
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred_my, label="My SVR", alpha=0.6)
plt.scatter(y_test, y_pred_sk, label="Sklearn SVR", alpha=0.6)
plt.scatter(y_test, y_pred_lr, label="Linear Reg", alpha=0.6)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.title("Model Comparison")
plt.show()
