import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Part 1: Data loading and preprocessing
#Load Data
df=pd.read_csv("/Users/mac/Desktop/Machine Learning/datasets/housing.csv")

#Preview
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

#Handle categorical variable
binary_cols=[
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]

for col in binary_cols:
    df[col]=df[col].map({"yes":1 , "no":0})
    
# one-hot encoding
df=pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

#checking missing values
missing_per=df.isnull().mean()*100
print("\nPercentage of missing values in each column:")
print(missing_per)

# drop columns with more than 80% missing values
columns_to_drop=missing_per[missing_per>80].index
df.drop(columns=columns_to_drop, inplace=True)

#drop rows with missing values is less than 5%
low_missing_cols = missing_per[(missing_per > 0) & (missing_per < 5)].index
df.dropna(subset=low_missing_cols, inplace=True)

#fill missing values in bedrooms with median
df["bedrooms"].fillna(df["bedrooms"].median(), inplace=True)

#Feature scalling
features=df.drop("price", axis=1)
target=df["price"]

scaler=StandardScaler()
features_scaled=scaler.fit_transform(features)

X=pd.DataFrame(features_scaled, columns=features.columns)
Y=target

#Part 2:Build Multiple Linear Regression Model

#Split data into training and testing sets(80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Model implementation
model=LinearRegression()
model.fit(X_train, Y_train)

#coefficients and intercept
print("Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef}")
    
print("Intercept:", model.intercept_)

#Manual calculation of coefficients and intercept
X_b=np.c_[np.ones((X.shape[0], 1)), X]
beta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
print("\nManual Coefficients:")
for col, coef in zip(X.columns, beta[1:]):
    print(f"{col}: {coef}")
print("Manual Intercept:", beta[0])

#Part 3: Model Evaluation

#Predicting on the test set
y_pred=model.predict(X_test)
print("Predicted values:", y_pred)
print("Actual values:", Y_test.values)

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')  # Line for perfect predictions
plt.show()

#Metrics
mae=mean_absolute_error(Y_test, y_pred)
mse=mean_squared_error(Y_test, y_pred)
r2=r2_score(Y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

#Part 4: Visualize the Regression Line

#Correlation heatmap
corr=df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90) #type: ignore
plt.yticks(range(len(corr)), corr.columns) #type: ignore
plt.title("Correlation Heatmap")
plt.show()

#Predicted vs Actual values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')  # Line for perfect predictions
plt.show()

#Part 5: Prediction
new_house=pd.DataFrame({
    'area': 2400,
    'bedrooms': 4,
    'bathrooms': 3,
    'stories': 2,
    'mainroad': 1,
    'guestroom': 0,
    'basement': 1,
    'hotwaterheating': 0,
    'airconditioning': 1,
    'parking': 2,
    'prefarea': 1,
    'furnishingstatus_semi-furnished': 1,
    'furnishingstatus_unfurnished': 0
})

new_house_scaled=scaler.transform(new_house)
predicted_price=model.predict(new_house_scaled)
print("Predicted price for the new house:", predicted_price[0])
