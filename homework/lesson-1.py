import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Introduction to Machine Learning: Exploring the Titanic Dataset

# Load the Titanic dataset
titanic_data = pd.read_csv('/Users/mac/Desktop/Machine Learning/ai-roadmap/content/ml/introduction/data/titanic.csv')

# Explore the dataset
print(f"\n Rows : {titanic_data.shape[0]}")
print("Columns:", titanic_data.shape[1])

# Display basic information about the dataset
print("\n Column names:")
print(titanic_data.columns)

# Display summary statistics for numerical columns
print("\n Summary statistics for numerical columns:")
print(titanic_data.describe())
print("\n Data types:")
print(titanic_data.dtypes)

# Display the first few rows of the dataset
print("\n First 5 rows of the dataset:")
print(titanic_data.head())

# Mean , median, and mode of the 'Age' column
age_mean = titanic_data['Age'].mean()
age_median = titanic_data['Age'].median()
age_mode = titanic_data['Age'].mode()[0]
print(f"\n Mean Age: {age_mean}")
print(f" Median Age: {age_median}")
print(f" Mode Age: {age_mode}")

# Survival by gender
survival_by_gender = titanic_data.groupby('Sex')['Survived'].mean()
print("\n Survival rate by gender:")
print(survival_by_gender)

# Survival by class
survival_by_class = titanic_data.groupby('Pclass')['Survived'].mean()
print("\n Survival rate by class:")
print(survival_by_class)

# Visualize the distribution of ages
plt.figure(figsize=(10, 6))
plt.hist(titanic_data['Age'].dropna(), bins=30, edgecolor='black')
plt.title('Distribution of Ages on the Titanic')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualize gender distribution
plt.figure(figsize=(6, 4))
gender_counts = titanic_data['Sex'].value_counts()
plt.bar(gender_counts.index, gender_counts.values) #type: ignore
plt.title('Gender Distribution on the Titanic')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Visualize survival rates by class
plt.figure(figsize=(8, 5))
class_survival_counts = titanic_data.groupby('Pclass')['Survived'].mean()
plt.bar(class_survival_counts.index, class_survival_counts.values) #type: ignore
plt.title('Survival Rates by Class on the Titanic')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.xticks(class_survival_counts.index)
plt.show()

# Visualize survival rates by gender
plt.figure(figsize=(6, 4))
gender_survival_counts = titanic_data.groupby('Sex')['Survived'].mean()
plt.bar(gender_survival_counts.index, gender_survival_counts.values) #type: ignore
plt.title(' Survival Rates by Gender on the Titanic')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

#Remove duplicates
titanic_data = titanic_data.drop_duplicates()

#Handle missing values by filling them with the mean age
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

#Encode categorical variables
titanic_data['Sex']= titanic_data['Sex'].map({"male": 0, "female": 1})

#Survival rate by age group
titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'], bins=[0, 12, 18, 35, 60, np.inf], labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])
survival_by_age_group = titanic_data.groupby('AgeGroup')['Survived'].mean()
print("\n Survival rate by age group:")
print(survival_by_age_group)

#Visualize survival rates by age group
plt.figure(figsize=(10, 6))
age_group_survival_counts = titanic_data.groupby('AgeGroup')['Survived'].mean()
age_group_survival_counts.plot(kind='bar')
plt.title('Survival Rates by Age Group on the Titanic')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.show()

#Family size 
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']
survival_by_family_size = titanic_data.groupby('FamilySize')['Survived'].mean()
print("\n Survival rate by family size:")
print(survival_by_family_size)

#Correlation between family size and survival
corr = titanic_data.corr(numeric_only=True)
plt.figure()
plt.imshow(corr)
plt.title("Correlation Matrix")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90) #type: ignore
plt.yticks(range(len(corr.columns)), corr.columns) #type: ignore
plt.colorbar()
plt.show()

def simple_rule(row):
    if row["Sex"] == 1 and row["Pclass"]==1:
        return 1
    return 0

titanic_data["PredictedSurvival"] = titanic_data.apply(simple_rule, axis=1)