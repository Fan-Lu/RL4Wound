import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directory containing images
image_dir = '/Users/bearcbass/RL4Wound/data/MouseData/train/0'
image_paths = [
    '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/1/Day 1_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/2/Day 2_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/3/Day 3_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/4/Day 4_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/5/Day 5_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/6/Day 6_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/7/Day 7_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/8/Day 8_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/9/Day 9_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/10/Day 10_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/11/Day 11_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/12/Day 12_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/13/Day 13_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/14/Day 14_Y8-2-L.png',
    '/Users/bearcbass/RL4Wound/data/MouseData/train/15/Day 15_Y8-2-L.png'
]


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

    cluster_means = []
    for i in range(n_clusters):
        cluster_pixels = img_array_flat[labels == i]
        avg_intensity = np.mean(cluster_pixels)
        cluster_means.append(avg_intensity)

    sorted_clusters = np.argsort(cluster_means)

    new_labels = np.zeros_like(labels)
    for i, cluster in enumerate(sorted_clusters):
        new_labels[labels == cluster] = i

    reordered_clustered_img_array_flat = np.array([cluster_colors[label] for label in new_labels])
    clustered_img_array = reordered_clustered_img_array_flat.reshape(img_array.shape)

    wound_cluster = 1 # assuming this is the wound
    wound_area = np.sum(new_labels == wound_cluster)

    return img, clustered_img_array, wound_area

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


for image_path in image_paths:
    original_img, clustered_image, _ = apply_kmeans(image_path, n_clusters, cluster_colors)
    original_images.append(original_img)
    clustered_images.append(clustered_image)
    titles.append(os.path.basename(image_path))  # Extract file name for title

# Plot all images
plot_images_with_legend(original_images[:5], clustered_images[:5], titles[:5], cluster_colors)
plot_images_with_legend(original_images[5:10], clustered_images[5:10], titles[5:10], cluster_colors)
plot_images_with_legend(original_images[10:15], clustered_images[10:15], titles[10:15], cluster_colors)



# Display first 5 images for brevity
#plot_images_with_legend(original_images[:5], clustered_images[:5], titles[:5], cluster_colors)
print(original_images)
