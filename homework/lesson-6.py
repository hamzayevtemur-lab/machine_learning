import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#load dataset
df=pd.read_csv("/Users/mac/Desktop/Machine Learning/datasets/car_price.csv")

print("First 5 rows:", df.head())
print("Data information", df.info())
print("Data summary:", df.describe())

# Handle missing values
print(df.isnull().sum())
#There is no missing values

#Columns with unusual distribution or outliers
print("\nColumns with unusual distribution or outliers:")
for col in df.columns:
    print(f"\n{col} - Summary Statistics:")
    print(df[col].describe())
    print(f"Number of unique values in {col}: {df[col].nunique()}")

#Numerical features
df=df.select_dtypes(include=[np.number])

#Target
Y=df["price"]

# Features
X = df.drop(columns=["price", "car_ID"])


#Data spliting
X_train, X_test, y_train, y_test = train_test_split(
    X, Y.values, test_size=0.2, random_state=42
)

#Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize parameters
m, n = X_train_scaled.shape
weights = np.zeros(n)
bias = 0

#Linear Model + Cost Function (Error and Function)
# Cost function (MSE)
def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X.dot(w) + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

#Gradient Descent Implementation
def gradient_descent(X,y,w,b, learning_rate, iterations):
    m=len(y)
    cost_history=[]
    
    for i in range(iterations):
        
        predictions = X.dot(w) + b
        
        #Gradients
        dw=(1/m)*X.T.dot(predictions-y)
        db=(1/m)*np.sum(predictions-y)
        
        #update
        w=w-learning_rate*dw
        b=b-learning_rate*db
        
        #Save cost
        cost=compute_cost(X,y,w,b)
        cost_history.append(cost)
        
    return w,b, cost_history


# Train Model
learning_rate=0.0001
iterations=1000

weights, bias, cost_history=gradient_descent(
    X_train_scaled, y_train, weights, bias, learning_rate, iterations
)

print("Final Cost:", cost_history[-1])

#Visualization (Convergence)
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

#Model Evaluation

#Predictions
X_test_scaled=scaler.transform(X_test)

y_pred_train=X_train_scaled.dot(weights)+bias
y_pred_test=X_test_scaled.dot(weights)+bias

# Metrics
print("TRAIN MSE:", mean_squared_error(y_train, y_pred_train))
print("TEST MSE:", mean_squared_error(y_test, y_pred_test))

print("TRAIN R2:", r2_score(y_train, y_pred_train))
print("TEST R2:", r2_score(y_test, y_pred_test))

# Predictions vs Actual
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")
plt.show()

#Feature Visualization (example: horsepower)
if "horsepower" in df.columns:
    plt.scatter(df["horsepower"], df["price"])
    plt.xlabel("Horsepower")
    plt.ylabel("Price")
    plt.title("Horsepower vs Price")
    plt.show()
    

### Bonus Parts
def mini_batch_gd(X,y, w, b, lr, iterations, batch_size):
    m=len(y)
    cost_history=[]
    
    for _ in range(iterations):
        for i in range(0, m, batch_size):
            X_batch=X[i:i+batch_size]
            y_batch=y[i:i+batch_size]
            
            predictions=X_batch.dot(w)+b
            
            dw=(1/len(y_batch))*X_batch.T.dot(predictions-y_batch)
            db=(1/len(y_batch))*np.sum(predictions-y_batch)
            
            w-=lr*dw
            b-=lr*db
            
        cost_history.append(compute_cost(X, y,w,b))
        
    return w,b, cost_history

model=LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred_sklearn=model.predict(X_test_scaled)

print("Sklearn MSE:", mean_squared_error(y_test, y_pred_sklearn))
print("Sklearn R2:", r2_score(y_test, y_pred_sklearn))


