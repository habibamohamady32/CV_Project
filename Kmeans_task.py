"""
K-means steps
    1-Select K random points as initial centroids
    2-Count distances between points in dataset and centroids
    3-Assign each point to its closest centroid
    4-Find each group of points and set it as new centroids
    5-Check if centroids moved. If no then repeat step 2 and 5, if yes then algorithm is covered, we found centroids for the given data
"""
import numpy as np
import random

def InitialCentroids(data, k):
    """
    Function picks k random data points from dataset.
    Returns:
     array of k unique initial centroids.
    """
    num_of_samples = data.shape[0]
    sample_points_id = random.sample(range(0, num_of_samples), k)
    centroids = [tuple(data[id]) for id in sample_points_id]
    unique_centroids = list(set(centroids))
    number_of_unique_centroids = len(unique_centroids)
    while number_of_unique_centroids < k:
        new_sample_points_ids = random.sample(range(0, num_of_samples), k - number_of_unique_centroids)
        new_centroids = [tuple(data[id]) for id in new_sample_points_ids]
        unique_centroids = list(set(unique_centroids + new_centroids))
        number_of_unique_centroids = len(unique_centroids)
    return np.array(unique_centroids)
def EuclideanDistance(matrix1, matrix2):
    """
    Function computes euclidean distance between matrix A and B.
    """
    matrix1_square = np.reshape(np.sum(matrix1 * matrix1, axis=1), (matrix1.shape[0], 1))
    matrix2_square = np.reshape(np.sum(matrix2 * matrix2, axis=1), (1, matrix2.shape[0]))
    result = matrix1@matrix2.T
    dst = -2 * result + matrix2_square + matrix1_square
    return np.sqrt(dst)
def getClusters(data, centroids):
    """
    Function finds k centroids and assigns each of the point of array data to the closest centroid
    Returns:
        list_of_points_in_cluster
    """
    clusters = {}
    distance_matrix = EuclideanDistance(data, centroids)
    closest_cluster_ids = np.argmin(distance_matrix, axis=1)
    for i in range(centroids.shape[0]):
        clusters[i] = []

    for i, cluster_id in enumerate(closest_cluster_ids):
        clusters[cluster_id].append(data[i])

    return clusters
def centroids_covered(previous_centroids, new_centroids, movement_delta):
    """
    Function checks if any of centroids are stand still then centroids were founded
        ///movement_delta: value, if centroids move less we assume that algorithm covered///
    Returns: boolean True if centroids coverd False if not
    """
    dist_between_old_and_new = EuclideanDistance(previous_centroids, new_centroids)
    centroidsCovered = np.max(dist_between_old_and_new.diagonal()) <= movement_delta
    return centroidsCovered
def Kmeans_algorithm(data, k, movement_delta=0):
    """
    Function performs k-means algorithm on a given dataset, finds and returns k centroids
    """
    newCentroids = InitialCentroids(data=data, k=k)
    centroidsCovered = False
    while not centroidsCovered:
        previousCentroids = newCentroids
        clusters = getClusters(data, previousCentroids)
        newCentroids = np.array([np.mean(clusters[key], axis=0, dtype=data.dtype) for key in sorted(clusters.keys())])
        centroidsCovered = centroids_covered(previousCentroids, newCentroids, movement_delta)
    return newCentroids
def reducedColors_image(image, number_of_colors):
    """
    Function returns given image with reduced number of colors
    """
    hight, width, dim = image.shape
    data = np.reshape(image, (hight * width, dim))
    data = np.array(data, dtype=np.int32)
    centroid = Kmeans_algorithm(data, k=number_of_colors, movement_delta=4)
    distance_matrix = EuclideanDistance(data, centroid)
    closest_cluster_id = np.argmin(distance_matrix, axis=1)
    reconstructed_data = centroid[closest_cluster_id]
    reconstructed_data = np.array(reconstructed_data, dtype=np.uint8)
    reducedImage = np.reshape(reconstructed_data, (hight, width, dim))
    return reducedImage


