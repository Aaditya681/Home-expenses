import numpy as np
from random import sample

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-Means Clustering Implementation
def k_means_clustering(data, k=3, max_iterations=100):
    # Randomly initialize centroids
    initial_centroids_idx = sample(range(len(data)), k)
    centroids = data[initial_centroids_idx]

    for _ in range(max_iterations):
        # Assign data points to closest centroid
        clusters = {}
        for x in data:
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            if closest_centroid in clusters:
                clusters[closest_centroid].append(x)
            else:
                clusters[closest_centroid] = [x]

        # Update centroids
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters.values()]

        # Check for convergence
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

# Function to calculate the Dunn Index
def dunn_index(centroids, clusters):
    if len(clusters) < 2:
        return 0

    # Calculate inter-cluster distances
    inter_cluster_distances = [euclidean_distance(centroids[i], centroids[j]) 
                               for i in range(len(centroids)) for j in range(i+1, len(centroids))]
    min_inter_cluster_distance = min(inter_cluster_distances)

    # Calculate intra-cluster distances
    max_intra_cluster_distance = max(
        [euclidean_distance(point, centroids[cluster_id]) 
         for cluster_id, cluster in clusters.items() for point in cluster]
    )

    return min_inter_cluster_distance / max_intra_cluster_distance

# Function to calculate inertia
def calculate_inertia(centroids, clusters):
    inertia = sum(
        [euclidean_distance(point, centroids[cluster_id]) ** 2 
         for cluster_id, cluster in clusters.items() for point in cluster]
    )
    return inertia

#load data
data = np.loadtxt('DMV302_Assessment_2_HouseholdWealth.csv', delimiter=',')
 

 
# Running K-Means for K=2 to K=10 and calculating Dunn index and inertia
dunn_indices = []
inertias = []
for k in range(2, 11):
    centroids, clusters = k_means_clustering(data, k=k, max_iterations=100)
    dunn_idx = dunn_index(centroids, clusters)
    inertia = calculate_inertia(centroids, clusters)
    dunn_indices.append(dunn_idx)
    inertias.append(inertia)

# Printing results
for k in range(2, 11):
    print(f"K={k}: Dunn Index = {dunn_indices[k-2]}, Inertia = {inertias[k-2]}")
