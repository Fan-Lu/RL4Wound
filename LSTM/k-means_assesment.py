import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and preprocess the image
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-1-R.png'

image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-4-L.png'
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_Y8-2-L.png'

# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-1-R.png'



image = cv2.imread(image_path)

# Apply bilateral filtering to the image (to smooth the image while preserving edges)
# d: Diameter of each pixel neighborhood that is used during filtering
# sigmaColor: Filter sigma in the color space
# sigmaSpace: Filter sigma in the coordinate space (spatial distance)
bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Reshape the filtered image into a 2D array of pixels (rows = number of pixels, columns = 3 for RGB)
pixels = bilateral_filtered_image.reshape(-1, 3)

# Perform K-means clustering
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

cluster_means = []
for i in range(k):
    cluster_pixels = pixels[kmeans.labels_ == i]
    avg_intensity = np.mean(cluster_pixels)
    cluster_means.append(avg_intensity)

sorted_clusters = np.argsort(cluster_means)

new_labels = np.zeros_like(kmeans.labels_)
for i, cluster in enumerate(sorted_clusters):
    new_labels[kmeans.labels_ == cluster] = i

# This section is to get the mask for contours

cluster_index = 1
mask = (new_labels == cluster_index)

mask_image = mask.reshape(image.shape[:2]).astype(np.uint8) * 255

contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(contour_image, contours, -1, (0,255,0),2)

filled_mask = np.zeros_like(mask_image)
for contour in contours:
    cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

refined_image = cv2.bitwise_and(image, image, mask = filled_mask)

# Replace each pixel value with its cluster center
clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]
clustered_image = clustered_pixels.reshape(image.shape).astype(np.uint8)

area_pixels = np.sum(filled_mask > 0)

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

# Visualize the results
plt.figure(figsize=(15, 10))

# Bilateral Filtered Image
plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Bilateral Filtered Image')

# Mask Image
plt.subplot(1, 4, 3)
plt.imshow(mask_image, cmap='gray')
plt.title(f'Cluster {cluster_index} Mask (Darkest Cluster)')

# Refined Image
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(refined_image, cv2.COLOR_BGR2RGB))
plt.title('Refined Image')

plt.show()