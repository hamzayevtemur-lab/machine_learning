import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB, MultinomialNB


#Load dataset
data_iris=load_iris()
X_iris=data_iris.data #type:ignore
y_iris=data_iris.target #type:ignore

# Train -test split
X_train_iris, X_test_iris, y_train_iris, y_test_iris=train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

#Initialize model
model=GaussianNB()

#Train
model.fit(X_train_iris, y_train_iris)

#Predictions
y_pred_iris=model.predict(X_test_iris)

print("Accuracy:", accuracy_score(y_test_iris, y_pred_iris))
print("\nClassification Report:\n", classification_report(y_test_iris, y_pred_iris))



##################### Multinomial Naive Bayes

#Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data=pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

#Convert labless
data["label"]=data["label"].map({"ham":0, "spam":1})

#Word frequency features
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(data["message"])
y=data["label"]

X_train, X_test, y_train, y_test=train_test_split(
    X,y, test_size=0.3, random_state=42
)

#Train model
multinomial=MultinomialNB()
multinomial.fit(X_train, y_train)

#Predictions
y_pred=multinomial.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))