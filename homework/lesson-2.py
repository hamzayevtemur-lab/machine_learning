import os
#import tarfile
#import urllib.request
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

#DOWNLOAD_ROOT = "https://github.com/ageron/data/raw/main/"
#HOUSING_PATH = os.path.join("datasets", "housing")
#HOUSING_URL = DOWNLOAD_ROOT + "housing.tgz"

#def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
#    os.makedirs(housing_path, exist_ok=True)
#   tgz_path = os.path.join(housing_path, "housing.tgz")
#    urllib.request.urlretrieve(housing_url, tgz_path)                          # After downloading the file, we can commit this function
#    with tarfile.open(tgz_path) as housing_tgz:
#        housing_tgz.extractall(path=housing_path)

#fetch_housing_data()

df = pd.read_csv("/Users/mac/Desktop/Machine Learning/datasets/housing/housing/housing.csv")


# Part 1: Exploratory Data Analysis (EDA)

#Display basic information about the dataset
print("Dataset Information:")
print("First 10 rows:")
print(df.head(10))
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())
print("Value Counts for 'ocean_proximity':")
print(df["ocean_proximity"].value_counts())

#Identify
#Columns with missing values
print("\nColumns with missing values:")
print(df.columns[df.isnull().any()])

#Numerical vs Categorical columns
numerical_cols=df.select_dtypes(include=['float64','int16']).columns
categorical_cols=df.select_dtypes(include=['object']).columns
print("\nNumerical Columns:")
print(numerical_cols)
print("\nCategorical Columns:")
print(categorical_cols)

#Columns with unusual distribution or outliers
print("\nColumns with unusual distribution or outliers:")
for col in numerical_cols:
    print(f"\n{col} - Summary Statistics:")
    print(df[col].describe())
    print(f"Number of unique values in {col}: {df[col].nunique()}") # nunique() counts the number of unique values in a column
    

# Part 2: Handling missing values

#drop rows or columns if missing values are insignificant
missing_percentage=df.isnull().mean()*100
print("\nPercentage of missing values in each column:")
print(missing_percentage)

# drop columns with more than 80% missing values
columns_to_drop=missing_percentage[missing_percentage>80].index
df.drop(columns=columns_to_drop, inplace=True)

#drop rows with missing values is less than 5%
low_missing_cols = missing_percentage[(missing_percentage > 0) & (missing_percentage < 5)].index
df.dropna(subset=low_missing_cols, inplace=True)

#median imputation for total_bedrooms
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

def missing_report(df):
    missing_count=df.isnull().sum()
    missing_percentage=(missing_count/len(df))*100
    
    report=pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    })
    
    return report[report["Missing Count"]>0]

print("\nMissing Values Report:")
print(missing_report(df))

# Part 3: Encoding Categorical Variables

#One-hot encoding for 'ocean_proximity'
df_encoded=pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
print("\nColumns after encoding:")
print(df_encoded.columns)

#Part 4: Feature Scaling
#Standardization for numerical features like median_income, housing_median_age, population, median_house_value

# StandartScaler()
scaler=StandardScaler()
features = ['median_income', 'housing_median_age', 'population', 'median_house_value']
X = df[features]
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=features)
print("\nFirst 5 rows of the scaled features:")
print(df_scaled.head())

# MinMaxScaler()
min_max_scaler=MinMaxScaler()
X_min_max_scaled=min_max_scaler.fit_transform(X)
df_min_max_scaled=pd.DataFrame(X_min_max_scaled, columns=features)
print("\nFirst 5 rows of the Min-Max scaled features:")
print(df_min_max_scaled.head())

#Plot feature histogram before and after scaling
for col in features:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(df[col], bins=30, edgecolor='black')
    plt.title(f'Original {col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(df_scaled[col], bins=30, edgecolor='black')
    plt.title(f'Standard Scaled {col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    plt.hist(df_min_max_scaled[col], bins=30, edgecolor='black')
    plt.title(f'Min-Max Scaled {col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
# Part 5: Optional Feature Engineering
# create new feature 'rooms_per_household'
df['rooms_per_household']=df['total_rooms']/df['households']
print("\nFirst 5 rows with new feature 'rooms_per_household':")
print(df[['total_rooms', 'households', 'rooms_per_household']].head())

# create new feature 'population_per_household'
df['population_per_household']=df['population']/df['households']
print("\nFirst 5 rows with new feature 'population_per_household':")
print(df[['population', 'households', 'population_per_household']].head())

# create new feature 'bedrooms_per_room'
df['bedrooms_per_room']=df['total_bedrooms']/df['total_rooms']
print("\nFirst 5 rows with new feature 'bedrooms_per_room':")
print(df[['total_bedrooms', 'total_rooms', 'bedrooms_per_room']].head())



