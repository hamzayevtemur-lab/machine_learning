import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge,Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv("/Users/mac/Desktop/Machine Learning/ai-roadmap/content/ml/supervised/04 regularisation/data/california_housing.csv")

#Preview
print("First 5 rows:", df.head())
print("Information:", df.info())
print("Summary:", df.describe())

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

#Unique Values
print("\nColumns with unusual distribution or outliers:")
for col in df.columns:
    print(f"Number of unique values in {col}: {df[col].nunique()}")
    
#Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

#Splitting Data
X=df.drop("MedHouseVal", axis=1)
y=df["MedHouseVal"]

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42)

#Sacel After splitting
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Baseline: Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

y_test_pred_lr=lin_reg.predict(X_test_scaled) 

mse_lr=mean_squared_error(y_test, y_test_pred_lr)
r2_lr=r2_score(y_test, y_test_pred_lr)

print("Linear Regression MSE:", mse_lr)
print("Linear Regression R2:", r2_lr)



########## Ridge Regression (L2 Regularization) ###########

alphas=np.logspace(-3, 3, 50)

best_alpha=None
best_mse=float("inf")
best_model=None 

# Loop over all alpha values
for alpha in alphas:
    model=Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    y_pred=model.predict(X_test_scaled)
    mse_l2=mean_squared_error(y_test, y_pred)
    
    #Select best model (lower MSE)
    if mse_l2<best_mse:
        best_mse=mse_l2
        best_alpha=alpha
        best_model=model
        

#Final evelution
y_pred_ridge=best_model.predict(X_test_scaled) #type:ignore
r2_ridge=r2_score(y_test, y_pred_ridge)

print("Best alpha (manual):", best_alpha)
print("Ridge MSE:", best_mse)
print("Ridge R2:", r2_ridge)

#Ridge Coefficient Plot
coefs=[]

for a in alphas:
    model=Ridge(alpha=a)
    model.fit(X_train_scaled, y_train)
    coefs.append(model.coef_)
    
plt.figure(figsize=(10,6))
plt.plot(alphas, coefs)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Ridge Coefficients vs Alpha")
plt.show()


################Lasso Regression (L1 Regularization)################

best_alpha_l1 = None
best_mse_l1 = float("inf")
best_model_l1 = None

for alpha in alphas:
    model_l1=Lasso(alpha=alpha, max_iter=10000)
    model_l1.fit(X_train_scaled, y_train)
    
    y_pred=model_l1.predict(X_test_scaled)
    mse_l1=mean_squared_error(y_test, y_pred)
    
    if mse_l1<best_mse_l1:
        best_mse_l1=mse_l1
        best_alpha_l1=alpha
        best_model_l1=model_l1
        
# Final evaluation
y_pred_lasso = best_model_l1.predict(X_test_scaled)   #type:ignore
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Best alpha (manual Lasso):", best_alpha_l1)
print("Lasso MSE:", best_mse_l1)
print("Lasso R2:", r2_lasso)

# Lasso Coefficient Plot
coefs_lasso = []

for a in alphas:
    model_l1 = Lasso(alpha=a, max_iter=10000)
    model_l1.fit(X_train_scaled, y_train)
    coefs_lasso.append(model_l1.coef_)

plt.figure(figsize=(10,6))
plt.plot(alphas, coefs_lasso)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Lasso Coefficients vs Alpha")
plt.show()


#Models Comparison
print("\n--- Model Comparison ---")

print(f"Linear Regression -> MSE: {mse_lr:.4f}, R2: {r2_lr:.4f}")
print(f"Ridge Regression  -> MSE: {best_mse:.4f}, R2: {r2_ridge:.4f}")
print(f"Lasso Regression  -> MSE: {best_mse_l1:.4f}, R2: {r2_lasso:.4f}")

# Count non-zero coefficients (Lasso feature selection)
non_zero_lasso = np.sum(best_model_l1.coef_ != 0) #type:ignore
print("Non-zero coefficients (Lasso):", non_zero_lasso)


####### Bonus Code (ElasticNet)

l1_ratios = [0.1, 0.5, 0.9]
best_alpha_net = None
best_l1 = None
best_mse_net = float("inf")
best_model_net = None

for alpha in alphas:
    for l1 in l1_ratios:
        model_net=ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000)
        model_net.fit(X_train_scaled, y_train)
        
        y_pred=model_net.predict(X_test_scaled)
        mse_net=mean_squared_error(y_test, y_pred)
        
        if mse_net < best_mse_net:
            best_mse_net = mse_net
            best_alpha_net = alpha
            best_l1 = l1
            best_model_net = model_net
            
            
            
y_pred = best_model_net.predict(X_test_scaled) #type:ignore

print("Best alpha:", best_alpha_net)
print("Best l1_ratio:", best_l1)
print("ElasticNet MSE:", mean_squared_error(y_test, y_pred))
print("ElasticNet R2:", r2_score(y_test, y_pred)) 