import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and preprocess the image - I have a collection of some just to experment with

# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-1-R.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-3-L.png'
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-3-R.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-4-L.png'
# image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-4-R.png'

image = cv2.imread(image_path)

# First im using bilateral filtering to try and remove the glare
bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Next we have to reshape the filtered image into a 2D array of pixels
pixels = bilateral_filtered_image.reshape(-1, 3)

# Perform K-means clustering on the 2d array
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

# Create a mask for the cluster of interest
cluster_index = 0
mask = (kmeans.labels_ == cluster_index)

# Reshape the mask to match the original image dimensions
mask_image = mask.reshape(image.shape[:2]).astype(np.uint8) * 255  # Binary mask (255 for cluster of interest)

# Find contours in the mask
contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and fill the detected regions
contour_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Draw contours (green)

# Fill the contours to address gaps
filled_mask = np.zeros_like(mask_image)
for contour in contours:
    cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

# Combine filled mask with original image
refined_image = cv2.bitwise_and(image, image, mask=filled_mask)

# Calculate the area of the mask
# Count non-zero pixels in the filled mask
area_pixels = np.sum(filled_mask > 0)

# Convert area from pixels to a more interpretable unit (e.g., square centimeters) if needed
# Assume each pixel represents a square unit of area; adjust as needed
pixel_area = 1  # Adjust this if you know the physical size of each pixel
area_units = area_pixels * pixel_area

print(f'Area of the mask (in pixels): {area_pixels}')
print(f'Area of the mask (in square units): {area_units}')

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
plt.title(f'Cluster {cluster_index} Mask')

# Refined Image
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(refined_image, cv2.COLOR_BGR2RGB))
plt.title('Refined Image')

plt.show()