import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples


# Load Dataset
X, y=make_moons(n_samples=500, noise=0.08, random_state=42)

# Standardize Features
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

# DBSCAN Model
dbscan=DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X_scaled)

labels=dbscan.labels_

# Number of clusters excluding noise
n_clusters=len(set(labels))-(1 if -1 in labels else 0) 

#Number of noise points
n_noise=list(labels).count(-1)

print("labels:")
print(labels)

print("\nNumber of clusters:", n_clusters)
print("Number of noise points:", n_noise)


# Visualization
unique_labels=set(labels)

plt.figure(figsize=(8, 6))

for label in unique_labels:
    
    # Noise points
    if label==-1:
        color="black"
        label_name="Noise"
    else:
        color=None
        label_name=f"Cluster {label}"
        
    cluster_points=X_scaled[labels==label]
    
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        c=color,
        label=label_name
    )
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


# Hyperparameter Tuning
configs = [
    (0.2, 5),
    (0.3, 5),
    (0.5, 5),
    (0.3, 10),
]

print("\n===== Hyperparameter Experiments =====")

for eps, min_samples in configs:
    model=DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X_scaled)
    
    labels=model.labels_
    
    n_clusters=len(set(labels))-(1 if -1 in labels else 0)
    n_noise=list(labels).count(-1)
    
    print(f"\neps={eps}, min_samples={min_samples}")
    print("Clusters:", n_clusters)
    print("Noise points:", n_noise)
    

## Compare with KMeans

kmeans=KMeans(n_clusters=2, random_state=42)
kmeans_labels=kmeans.fit_predict(X_scaled)

centroids=kmeans.cluster_centers_

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=50
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    marker="X",
    s=300,
    label="Centroids"
)

plt.title("KMeans Clustering with Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show() 
