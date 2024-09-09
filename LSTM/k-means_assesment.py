import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and preprocess the image
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-1-R.png'
image = cv2.imread(image_path)

# Apply bilateral filtering to the image (to smooth the image while preserving edges)
# d: Diameter of each pixel neighborhood that is used during filtering
# sigmaColor: Filter sigma in the color space
# sigmaSpace: Filter sigma in the coordinate space (spatial distance)
bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Reshape the filtered image into a 2D array of pixels (rows = number of pixels, columns = 3 for RGB)
pixels = bilateral_filtered_image.reshape(-1, 3)

# Perform K-means clustering
k = 4  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

# Replace each pixel value with its cluster center
clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]
clustered_image = clustered_pixels.reshape(image.shape).astype(np.uint8)

# Visualize the original, filtered, and clustered images
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Bilateral Filtered Image
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Bilateral Filtered Image')

# K-Means Clustered Image
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(clustered_image, cv2.COLOR_BGR2RGB))
plt.title(f'K-Means Clustered Image with {k} Clusters')

plt.show()

# Visualize each cluster separately
plt.figure(figsize=(15, 10))
for i in range(k):
    # Create a mask for each cluster
    mask = (kmeans.labels_ == i)
    
    # Create a new image with only the pixels of the current cluster
    cluster_image = np.zeros_like(pixels)
    cluster_image[mask] = pixels[mask]  # Retain original color for the cluster

    # Reshape the cluster image back to original dimensions
    cluster_image = cluster_image.reshape(image.shape)

    # Plot the cluster
    plt.subplot(1, k, i + 1)
    plt.imshow(cv2.cvtColor(cluster_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Cluster {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()