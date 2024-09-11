import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# K-means clustering function
def apply_kmeans(image, n_clusters=3):
    img_array = image.reshape(-1, 3)  # Flatten the image to (n_pixels, 3)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_array)
    labels = kmeans.labels_.reshape(image.shape[:2])  # Reshape back to image size
    
    return labels, kmeans

# Load the image
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/2/Day 2_Y8-2-L.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# First K-means clustering
n_clusters = 3
labels, kmeans = apply_kmeans(image_rgb, n_clusters=n_clusters)

# Visualize the clustered image
clustered_img = np.zeros_like(image_rgb)
for cluster in range(n_clusters):
    clustered_img[labels == cluster] = kmeans.cluster_centers_[cluster]

plt.imshow(clustered_img)
plt.title('First K-means: Clustered Image')
plt.show()

# Assume you want to work on a specific cluster (e.g., cluster 1 after reordering)
wound_cluster = 0  # Chosen cluster

# Create a mask for the selected cluster
selected_cluster_mask = (labels == wound_cluster).astype(np.uint8) * 255

# Use the mask to isolate the selected cluster's pixels in the original image
isolated_cluster_image = np.zeros_like(image_rgb)
isolated_cluster_image[selected_cluster_mask == 255] = image_rgb[selected_cluster_mask == 255]

plt.imshow(isolated_cluster_image)
plt.title(f'Isolated Cluster {wound_cluster}')
plt.show()

# Apply second K-means clustering on the isolated pixels
non_zero_pixels = isolated_cluster_image[selected_cluster_mask == 255].reshape(-1, 3)  # Flatten the non-zero pixels
n_clusters_second = 2  # Number of clusters for second K-means
kmeans_second = KMeans(n_clusters=n_clusters_second, random_state=0).fit(non_zero_pixels)

# Map the second k-means result back to the original image size
labels_second = np.zeros(image_rgb.shape[:2], dtype=int)
labels_second[selected_cluster_mask == 255] = kmeans_second.labels_ + 1  # Avoid 0 for background

# Function to isolate and visualize a single cluster from second k-means
def visualize_single_cluster(image, labels, target_cluster):
    isolated_cluster = np.zeros_like(image)
    isolated_cluster[labels == target_cluster] = image[labels == target_cluster]
    return isolated_cluster

# Visualize each cluster from second k-means
for cluster_id in range(1, n_clusters_second + 1):
    cluster_image = visualize_single_cluster(image_rgb, labels_second, cluster_id)
    plt.imshow(cluster_image)
    plt.title(f'Second K-means: Isolated Cluster {cluster_id}')
    plt.show()
