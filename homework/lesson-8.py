import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE

df=pd.read_csv("/Users/mac/Desktop/Machine Learning/ai-roadmap/content/ml/supervised/05 logistic regression/data/heart_diceases.csv")

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
    
    
########Visualization
#Histogram
df.hist(figsize=(12,10))
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()



########Data Preprocessing
#One-Hot Encoding
df=pd.get_dummies(df, columns=['cp', 'slope', 'thal'], drop_first=True)

#Split features and target
X=df.drop("target")
y=df["target"]

#Train-Test Split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42)

#Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

###### Model Building
log_reg=LogisticRegression(solver="liblinear")
log_reg.fit(X_train_scaled, y_train)

##### Model Evaluation

#prediction
y_pred=log_reg.predict(X_test)
y_prob=log_reg.predict_proba(X_test)[:,1]

#Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

#Classification report
print(classification_report(y_test, y_pred))

#Confusion matrix
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

#ROC curve
fpr, tpr, _=roc_curve(y_test, y_prob)
roc_auc=auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


##### Feature Importance
coefficients=pd.DataFrame({
    "Feature": X.columns,
    "Coefficient":log_reg.coef_[0]
    })

print(coefficients.sort_values(by="Coefficient", ascending=False))


#RFE (Feature Selection)

selector=RFE(log_reg, n_features_to_select=8)
selector.fit(X_train_scaled, y_train)

selected_features=X.columns[selector.support_]
print(selected_features)


##### Prediction Example
sample_dict = {
    'age': 63,
    'sex': 1,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'ca': 0,

    'cp_1': 0,
    'cp_2': 0,
    'cp_3': 1,

    'slope_1': 0,
    'slope_2': 1,

    'thal_1': 0,
    'thal_2': 1,
    'thal_3': 0
}

# Convert to DataFrame
sample = pd.DataFrame([sample_dict])

# Match column order EXACTLY
sample = sample[X.columns]

# Scale
sample_scaled = scaler.transform(sample)

# Predict
prediction = log_reg.predict(sample_scaled)
probability = log_reg.predict_proba(sample_scaled)

print("Heart Disease:", "YES" if prediction[0]==1 else "NO")



####  BONUS
#Hyperparameter Tuning

params={
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid=GridSearchCV(LogisticRegression(), params, cv=5)
grid.fit(X_train_scaled, y_train)

print("Best Params:", grid.best_params_)








