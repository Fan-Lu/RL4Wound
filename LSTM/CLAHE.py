import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = cv2.imread('/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-1-R.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Convert the image to LAB color space for better luminance control
lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# Apply CLAHE to the L-channel (lightness)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l, a, b = cv2.split(lab_image)
l_clahe = clahe.apply(l)

# Merge the channels back together
lab_clahe_image = cv2.merge((l_clahe, a, b))

# Convert back to RGB
clahe_image = cv2.cvtColor(lab_clahe_image, cv2.COLOR_LAB2RGB)

# Apply Gaussian blur to reduce glare
blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)

# Reshape the image into a 2D array of pixels (rows = number of pixels, columns = 3 for RGB)
pixels = blurred_image.reshape(-1, 3)

# Perform K-means clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

# Replace each pixel value with its cluster center
clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]
clustered_image = clustered_pixels.reshape(image.shape).astype(np.uint8)

# Visualize each cluster separately
plt.figure(figsize=(15, 10))
for i in range(k):
    # Create a mask for the current cluster
    mask = (kmeans.labels_ == i)
    
    # Create an image for the current cluster, keeping only the pixels that belong to the cluster
    cluster_image = np.zeros_like(pixels)
    cluster_image[mask] = pixels[mask]  # Retain the original pixel values for the current cluster

    # Reshape the cluster image back to its original dimensions
    cluster_image = cluster_image.reshape(image.shape)

    # Plot the current cluster
    plt.figure()
    plt.imshow(cluster_image)
    plt.title(f'Cluster {i + 1}')
    plt.axis('off')

plt.show()