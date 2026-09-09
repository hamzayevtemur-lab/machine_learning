import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_diabetes

### **1. Data Loading and Exploration**
data=load_diabetes()

X=pd.DataFrame(data.data, columns=data.feature_names) #type: ignore
Y=pd.Series(data.target, name="target") #type: ignore

df=pd.concat([X,Y], axis=1)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("Feature names:", data.feature_names) #type: ignore

# 2. POLYNOMIAL FEATURES

features_to_transform=['bmi', 'bp', 's5']    # Select 3 features for polynomial transformation
poly=PolynomialFeatures(degree=2, include_bias=False)

X_poly_selected=poly.fit_transform(X[features_to_transform])

new_feature_names=poly.get_feature_names_out(features_to_transform)  # Get new feature names after polynomial transformation
print("Number of new features after polynomial transformation:", len(new_feature_names))
print("\nNew feature names after polynomial transformation:")
print(new_feature_names)

keep_features=[col for col in X.columns if col not in features_to_transform]  # Keep the remaining features unchanged

X_final=pd.concat([X[keep_features], pd.DataFrame(X_poly_selected, columns=new_feature_names)], axis=1)
print("\nFinal feature matrix shape:", X_final.shape)
print("\nFirst 5 rows of the final feature matrix:")
print(X_final.head())

# 3. MODEL BUILDING
#Split the data into training and testing sets
X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

#Apply same split to the polynomial features
X_poly_train=X_final.loc[X_train.index]  # Get polynomial features for the training set
X_poly_test=X_final.loc[X_test.index]    # Get polynomial features for the testing set

#Baseline Linear Regression Model on original features
model_baseline=LinearRegression()
model_baseline.fit(X_train, y_train)
y_pred_train_baseline=model_baseline.predict(X_train)
print("\nBaseline Linear Regression Model R-squared on training set:", model_baseline.score(X_train, y_train))
model_baseline.score(X_train, y_train)

#Polynomial Regression Model on polynomial features
model_poly=LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_train_poly=model_poly.predict(X_poly_train)
print("\nPolynomial Regression Model R-squared on training set:", model_poly.score(X_poly_train, y_train))
model_poly.score(X_poly_train, y_train)

#Top 10 Coefficients in the polynomial regression model
coefficients=pd.Series(model_poly.coef_, index=X_poly_train.columns)
top_coefficients=coefficients.abs().sort_values(ascending=False).head(10)
print("\nTop 10 coefficients in the polynomial regression model:")
for feature, coef in top_coefficients.items():
    print(f"{feature}: {coef:.4f}")
    
# 4. MODEL EVALUATION
# Predictions
y_pred_baseline=model_baseline.predict(X_test)
y_pred_poly=model_poly.predict(X_poly_test)

def evaluate_model(y_true, y_pred, model_name):
    mae=mean_absolute_error(y_true, y_pred)
    mse=mean_squared_error(y_true, y_pred)
    r2=r2_score(y_true, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
evaluate_model(y_test, y_pred_baseline, "Baseline Linear Regression Model")
evaluate_model(y_test, y_pred_poly, "Polynomial Regression Model")

# 5. VISUALIZATION
# 1. Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_baseline, color='blue', label='Baseline Linear Regression')
plt.scatter(y_test, y_pred_poly, color='red', label='Polynomial Regression')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--')  # Line for perfect predictions
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()

# 2. Error Comparison
mae_linear = mean_absolute_error(y_test, y_pred_baseline)
mae_poly = mean_absolute_error(y_test, y_pred_poly)

rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

labels = ["MAE", "RMSE"]
linear_vals = [mae_linear, rmse_linear]
poly_vals = [mae_poly, rmse_poly]

x=np.arange(len(labels))

plt.figure()
plt.bar(x - 0.2, linear_vals, width=0.4, label="Linear")
plt.bar(x + 0.2, poly_vals, width=0.4, label="Polynomial")

plt.xticks(x, labels)
plt.title("Error Comparison")
plt.legend()
plt.show()

# 3.Polynomial Curve (BMI)

X_bmi=X[["bmi"]]

poly_bmi=PolynomialFeatures(degree=2, include_bias=False)
X_bmi_poly=poly_bmi.fit_transform(X_bmi)

model_bmi_poly=LinearRegression()
model_bmi_poly.fit(X_bmi_poly, Y)

#Smooth curve for BMI
sorted_idx=X_bmi["bmi"].argsort()
X_bmi_sorted=X_bmi.iloc[sorted_idx]
X_bmi_poly_sorted=poly_bmi.transform(X_bmi_sorted)

y_bmi_poly_pred=model_bmi_poly.predict(X_bmi_poly_sorted)

plt.figure(figsize=(10, 6))
plt.scatter(X_bmi, Y, color='blue', label='Data Points')
plt.plot(X_bmi_sorted, y_bmi_poly_pred, color='red', label='Polynomial Fit')
plt.xlabel("BMI")
plt.ylabel("Target")
plt.title("Polynomial Fit for BMI vs Target")
plt.legend()
plt.show()

# 6. NEW PREDICTION

new_data = pd.DataFrame([{
    "age": 0.05,
    "sex": 0.02,
    "bmi": 0.04,
    "bp": 0.03,
    "s1": -0.02,
    "s2": -0.01,
    "s3": 0.00,
    "s4": 0.02,
    "s5": 0.03,
    "s6": 0.01
}])

#Polynomial features for the new data point
new_data_poly = poly.transform(new_data[features_to_transform])
poly_df = pd.DataFrame(new_data_poly, columns=poly.get_feature_names_out(features_to_transform)) #type: ignore

new_data_final = pd.concat([new_data[keep_features], poly_df], axis=1)

#Linear regression prediction
predicted_value_linear = model_baseline.predict(new_data)
print("Predicted target value for the new data point using Linear Regression:", predicted_value_linear[0])

#Prediction using the polynomial regression model
predicted_value = model_poly.predict(new_data_final)
print("\nPredicted target value for the new data point using Polynomial Regression:", predicted_value[0])

