import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and preprocess the image
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/1/Day 1_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/2/Day 2_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/3/Day 3_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/4/Day 4_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/5/Day 5_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/6/Day 6_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/7/Day 7_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/8/Day 8_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/9/Day 9_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/10/Day 10_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/11/Day 11_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/12/Day 12_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/13/Day 13_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/14/Day 14_Y8-2-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/15/Day 15_Y8-2-L.png'





image = cv2.imread(image_path)

# Step 1: Apply bilateral filtering
bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Step 2: Reshape the filtered image into a 2D array of pixels
pixels = bilateral_filtered_image.reshape(-1, 3)

# Step 3: Perform K-means clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

# Step 4: Calculate the average pixel intensity for each cluster
cluster_means = []
for i in range(k):
    cluster_pixels = pixels[kmeans.labels_ == i]
    avg_intensity = np.mean(cluster_pixels)
    cluster_means.append(avg_intensity)

# Step 5: Sort clusters by average intensity (lowest to highest)
sorted_clusters = np.argsort(cluster_means)

# Step 6: Reassign the cluster labels to match the sorted order
new_labels = np.zeros_like(kmeans.labels_)
for i, cluster in enumerate(sorted_clusters):
    new_labels[kmeans.labels_ == cluster] = i

# Step 7: Create a mask for the cluster of interest (after reordering)
cluster_index = 1  # Now, the darkest cluster will always be 0
mask = (new_labels == cluster_index)

# Step 8: Reshape the mask to match the original image dimensions
mask_image = mask.reshape(image.shape[:2]).astype(np.uint8) * 255  # Binary mask (255 for cluster of interest)

# Step 9: Find contours in the mask
contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 10: Draw contours and fill the detected regions
contour_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Draw contours (green)

# Fill the contours to address gaps
filled_mask = np.zeros_like(mask_image)
for contour in contours:
    cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

# Combine filled mask with original image
refined_image = cv2.bitwise_and(image, image, mask=filled_mask)

# Calculate the area of the mask
area_pixels = np.sum(filled_mask > 0)

# Visualize the results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

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