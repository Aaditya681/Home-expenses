import numpy as np
import pandas as pd

from random import sample

def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))

def k_means_clustering(data, k=3, max_iterations=100):
    """K-Means clustering algorithm."""
    # Randomly initialize the centroids
    initial_centroids_idx = sample(range(len(data)), k)
    centroids = data[initial_centroids_idx]

    for _ in range(max_iterations):
        # Assign each data point to the closest centroid
        clusters = {}
        for x in data:
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            if closest_centroid in clusters:
                clusters[closest_centroid].append(x)
            else:
                clusters[closest_centroid] = [x]

        # Update centroids to be the mean of points in each cluster
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters.values()]

        # Check for convergence
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

def dunn_index(centroids, clusters):
    """Calculate the Dunn Index for the given clustering."""
    if len(clusters) < 2:
        return 0

    inter_cluster_distances = [euclidean_distance(centroids[i], centroids[j]) 
                               for i in range(len(centroids)) for j in range(i+1, len(centroids))]
    min_inter_cluster_distance = min(inter_cluster_distances)

    max_intra_cluster_distance = max(
        [euclidean_distance(point, centroids[cluster_id]) 
         for cluster_id, cluster in clusters.items() for point in cluster]
    )

    return min_inter_cluster_distance / max_intra_cluster_distance

def calculate_inertia(centroids, clusters):
    """Calculate inertia for the given clustering."""
    inertia = sum(
        [euclidean_distance(point, centroids[cluster_id]) ** 2 
         for cluster_id, cluster in clusters.items() for point in cluster]
    )
    return inertia

# Load data from CSV


data = pd.read_csv('DMV302_Assessment_2_HouseholdWealth.csv').to_numpy()

# Running K-Means for K=2 to K=10 and calculating Dunn index and inertia
results = []
for k in range(2, 11):
    centroids, clusters = k_means_clustering(data, k=k, max_iterations=100)
    dunn_idx = dunn_index(centroids, clusters)
    inertia = calculate_inertia(centroids, clusters)
    results.append((k, dunn_idx, inertia))

# Printing results
for k, dunn_idx, inertia in results:
    print(f"K={k}: Dunn Index = {dunn_idx}, Inertia = {inertia}")


