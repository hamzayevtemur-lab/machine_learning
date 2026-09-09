import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data=pd.read_csv(url, sep="\t", header=None, names=["label", "message"])


print("Dataset preview:")
print(data.head())

print("Dataset Info:")
print(data.info())

print("Dataset summary:")
print(data.describe())


# Convert text to binary features
data["label"]=data["label"].map({"ham":0, "spam":1})

#Convert text to binary features
vectorizer=CountVectorizer(binary=True)
X=vectorizer.fit_transform(data["message"])
y=data["label"]

#Train-test split 
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.3, random_state=42)


#Train Bernoulli Naive Bayes
model=BernoulliNB()
model.fit(X_train, y_train)


#Prediction and evaluation
y_pred=model.predict(X_test)


#Accuracy
accuracy=accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

#Confusion Matrix
cm=confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
