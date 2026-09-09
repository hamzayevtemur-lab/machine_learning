import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components=n_components
    
    def fit(self, X):
        # Compute mean
        self.mean_=np.mean(X, axis=0)
        
        # Center the data
        X_centered=X-self.mean_
        
        # Covariance matrix
        cov_matrix=np.cov(X_centered, rowvar=False)
        
        # Eigen decomposition
        eigenvalues, eigenvectors=np.linalg.eig(cov_matrix)
        
        # Sort eigenvalues descending
        sorted_idx=np.argsort(eigenvalues)[::-1]
        
        eigenvalues=eigenvalues[sorted_idx]
        eigenvectors=eigenvectors[:, sorted_idx]
        
        # Select topm components
        self.components_=eigenvectors[:, :self.n_components].T
        
        # Explained variance
        self.explained_variance_=eigenvalues[:self.n_components]
        
        # Explained variance ratio
        total_variance=np.sum(eigenvalues)
        
        self.explained_variance_ratio_=(
            self.explained_variance_/total_variance
        )
        
        return self
    
    def transform(self, X):
        # Center using training mean
        X_centered=X-self.mean_
        
        # Project data
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        self.fit(X)
        
        return self.transform(X)
    
    

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load dataset
data=load_breast_cancer()

X=data.data #type: ignore
y=data.target #type: ignore

## Standardize Features
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)


## Create PCA object
pca=PCA(n_components=2)

# Fit and transform
X_pca=pca.fit_transform(X_scaled)

# Results
print("Transformed Shape:", X_pca.shape)

print("\nComponents Shape:")
print(pca.components_.shape)

print("\nExplained Variance:")
print(pca.explained_variance_)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)


## Visualiza PCA Projection
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="coolwarm"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Manual PCA Projection")

plt.colorbar()

plt.show()

