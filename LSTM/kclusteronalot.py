import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directory containing images
image_dir = '/Users/bearcbass/RL4Wound/data/MouseData/train/0'

# Number of clusters
n_clusters = 3

# Define colors for each cluster
cluster_colors = {
    0: [255, 105, 180],   # Pink
    1: [0, 255, 0],       # Green
    2: [128, 0, 128]      # Purple
}

# Function to apply K-Means clustering to an image and assign specific colors to clusters
def apply_kmeans(image_path, n_clusters, cluster_colors):
    # Load image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is in RGB mode

    # Convert image to numpy array
    img_array = np.array(img)
    img_array_flat = img_array.reshape(-1, 3)  # Flatten to (n_pixels, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(img_array_flat)

    # Map cluster labels to colors
    clustered_img_array_flat = np.array([cluster_colors[label] for label in labels])
    clustered_img_array = clustered_img_array_flat.reshape(img_array.shape)

    return img, clustered_img_array

# Plot images with legends
def plot_images_with_legend(original_images, clustered_images, titles, cluster_colors):
    n_images = len(original_images)
    fig, axes = plt.subplots(n_images, 2, figsize=(10, 5 * n_images))
    
    for i in range(n_images):
        ax1, ax2 = axes[i]
        
        # Plot original image
        ax1.imshow(original_images[i])
        ax1.set_title(f'Original Image\n{titles[i]}')
        ax1.axis('off')
        
        # Plot clustered image
        ax2.imshow(clustered_images[i])
        ax2.set_title('Clustered Image')
        ax2.axis('off')
        
        # Create legend
        handles = [patches.Patch(color=np.array(cluster_colors[j])/255, label=f'Cluster {j}') for j in cluster_colors]
        ax2.legend(handles=handles, loc='upper right')

    plt.show()

# Process and plot images
original_images = []
clustered_images = []
titles = []

for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        original_img, clustered_image = apply_kmeans(image_path, n_clusters, cluster_colors)
        original_images.append(original_img)
        clustered_images.append(clustered_image)
        titles.append(filename)

# Display first 5 images for brevity
plot_images_with_legend(original_images[:3], clustered_images[:3], titles[:3], cluster_colors)
print(original_images)
