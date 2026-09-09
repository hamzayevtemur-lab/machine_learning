#Simple Linear Regression: Predicting House Prizes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('/Users/mac/Desktop/Machine Learning/datasets/housing.csv')

#Part 1: Data Exploration
#Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(data['area'], data['price'])
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Size vs Price')
plt.show()

#Checking Outliers
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.boxplot(data['area'])
plt.title('Boxplot of Size')

plt.subplot(1, 2, 2)
plt.boxplot(data['price'])
plt.title('Boxplot of Price')
plt.show()

#Part 2: Build a Simple Linear Regression Model
#Split data into training and testing sets(80% train, 20% test)
X=data[['area']] #Feature
Y=data['price'] #Target variable

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Manual Calculation

# Convert to numpy
x = X_train['area'].values
y = Y_train.values

# Means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate slope (m)
m = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)

# Calculate intercept (b)
b = y_mean - m * x_mean

print("Manual slope (m):", m)
print("Manual intercept (b):", b)

#Using Scikit-learn
model = LinearRegression()
model.fit(X_train, Y_train)

print("Scikit-learn slope (m):", model.coef_[0])
print("Scikit-learn intercept (b):", model.intercept_)

# Part 3: Model Evaluation
# Predicting on the test set
Y_pred = model.predict(X_test)
print("Predicted values:", Y_pred)
print("Actual values:", Y_test.values)

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

#Part 4: Visualize the Regression Line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Regression Line")
plt.show()

# Residual Plot
residuals = Y_test - Y_pred
plt.figure(figsize=(10, 6))
plt.scatter(Y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

#Part 5: Prediction
# New input (15000 sq ft)
new_area = np.array([[15000]])
# Predict using trained model
predicted_price = model.predict(new_area)
print("Predicted price for 15000 sq ft:", predicted_price[0])

#Checking the range of area and price in the dataset
print(data['area'].min(), data['area'].max())
print(data['price'].min(), data['price'].max())

#Visualize the prediction on the scatter plot
plt.scatter(data['area'], data['price'])
plt.scatter(15000, predicted_price, color='red')  # your prediction
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Prediction Check")
plt.show()
