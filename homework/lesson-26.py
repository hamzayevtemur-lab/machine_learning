import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris


class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.tol=tol
        self.random_state=random_state
        
        
    # Euclidean Distance
    def _compute_distances(self, X, centroids):
        distance=np.sqrt((X[:, np.newaxis]-centroids)**2).sum(axis=2)
        return distance
    
    # Fit method
    def fit(self, X):
        np.random.seed(self.random_state)
        
        n_samples, n_features=X.shape
        
        # randomly choose initail centroids
        random_indices=np.random.choice(
            n_samples, 
            self.n_clusters,
            replace=True
        )
        
        centroids=X[random_indices]
        
        for iteration in range(self.max_iter):
            
            # Compute distance
            distances=self._compute_distances(X, centroids)
            
            # Assign clusters
            labels=np.argmin(distances, axis=1)
            
            # Compute new centroids
            new_centroids=np.array([
                X[labels==k].mean(axis=0)
                for k in range(self.n_clusters)
            ])
            
            # Check convergence
            centroids_shift=np.linalg.norm(new_centroids-centroids)
            
            if centroids_shift<self.tol:
                break
            
            centroids=new_centroids
            
        # Final assignments
        final_distances=self._compute_distances(X, centroids)
        final_labels=np.argmin(final_distances, axis=1)
        
        # Inertia (WCSS)
        inertia=np.sum(
            (X-centroids[final_labels])**2
        )
        
        # Required attributes
        self.cluster_centers_=centroids
        self.labels_=final_labels
        self.inertia_=inertia
        self.n_iter_=iteration + 1 #type:ignore
        
        return self
    
    # Predict
    def predict(self, X):
         
        distance=self._compute_distances(
            X,
            self.cluster_centers_
        )       
        
        return np.argmin(distance, axis=1)
    
    
    # Fit and predict
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    

# Dataset
X, y = make_blobs( #type:ignore
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.5,
    random_state=42

)

kmeans=KMeans(
    n_clusters=3, 
    random_state=42
)

kmeans.fit(X)

## Print Results

print("Cluster Centers:\n")
print(kmeans.cluster_centers_)

print("\nInertia:")
print(kmeans.inertia_)

print("\nNumber of Iterations:")
print(kmeans.n_iter_)

# Visualization
X_plot=X[:, :2]

plt.figure(figsize=(8, 6))

# Plot clustered points
plt.scatter(
    X_plot[:, 0],
    X_plot[:, 1],
    c=kmeans.labels_,
    cmap="viridis",
    s=50
)

#Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=300,
    label="Centroids"
)

plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.show()





     
        