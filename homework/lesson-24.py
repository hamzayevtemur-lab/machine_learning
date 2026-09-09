import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data=load_breast_cancer()

X=data.data #type:ignore
y=data.target #type:ignore

df=pd.DataFrame(data=X)

# Dataset Information
print("Dataset Shape:", X.shape)
print("Number of Features:", X.shape[1])
print("First 5 rows:", df.head())
print("Info:", df.info())
print("Summary:", df.describe())


# Unique values
for col in df.columns:
    print(col, df[col].nunique())
    

# Split dataset
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
model=LogisticRegression(max_iter=10000)

#Measure training time
start=time.time()
model.fit(X_train, y_train)
end=time.time()

# Predictions
y_pred=model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Training Time:", end - start)


##### Feature Scaling

# Scale data
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

scaled_model=LogisticRegression(max_iter=10000)

#Measure training time
start=time.time()
scaled_model.fit(X_train_scaled, y_train)
end=time.time()

# Predictions
scaled_pred=scaled_model.predict(X_test_scaled)

# Metrics
scaled_accuracy = accuracy_score(y_test, scaled_pred)

print("Scaled Accuracy:", scaled_accuracy)
print("Scaled Training Time:", end - start)


### Apply PCA
from sklearn.decomposition import PCA

#PCA
pca=PCA(n_components=2)

X_pca=pca.fit_transform(X_train_scaled)

print("Original Shape:", X_train_scaled.shape)
print("Reduced Shape:", X_pca.shape)


### Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:,1],
    c=y_train,
    cmap="coolwarm"
)

plt.xlabel(data.feature_names[0]) #type:ignore
plt.ylabel(data.feature_names[1]) #type:ignore
plt.title("Original Feature Space")
plt.colorbar()

plt.show()

### PCA components Plot
plt.figure(figsize=(7, 5))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y_train,
    cmap="coolwarm"
)


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection")
plt.colorbar()

plt.show()


### PCA with all components
pca_full=PCA(n_components=None)
pca_full.fit(X_train_scaled)
explained_variance=pca_full.explained_variance_ratio_
print(explained_variance)

### Plot Explained Variance Ratio
plt.figure(figsize=(10, 5))

plt.bar(
    range(1, len(explained_variance)+1),
    explained_variance
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Component")

plt.show()


### Plot cumulative explained variance
cumulative_variance=np.cumsum(explained_variance)

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(cumulative_variance)+1),
    cumulative_variance, 
    marker="o"
)
plt.axhline(
    y=0.90,
    color="red",
    linestyle="--"
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")

plt.show()


# Components needed for 90% variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("Components needed for 90% variance:", n_components_90)



## Reduce to 90% variance
pca_90=PCA(n_components=0.90)

X_train_pca = pca_90.fit_transform(X_train_scaled)
X_test_pca = pca_90.transform(X_test_scaled)

print("Reduced Shape:", X_train_pca.shape)



## Train Logistic Regression PCA 
pca_model=LogisticRegression(max_iter=10000)

start=time.time()
pca_model.fit(X_train_pca, y_train)
end=time.time()

# Predictions
pca_pred=pca_model.predict(X_test_pca)

# Metrics
pca_accuracy=accuracy_score(y_test, pca_pred)

print("PCA Accuracy:", pca_accuracy)
print("PCA Training Time:", end - start)



### Apply t-SNE
from sklearn.manifold import TSNE

tsne=TSNE(
    n_components=2, 
    random_state=42,
    perplexity=30
)

X_tsne=tsne.fit_transform(X_train_scaled)


# plot t-SNE visualization
plt.figure(figsize=(7,5))
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_train,
    cmap='coolwarm'
)
plt.title("t-SNE Visualization")
plt.colorbar()
plt.show()