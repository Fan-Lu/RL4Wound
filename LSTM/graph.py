import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define colors for each cluster
cluster_colors = {
    0: [255, 105, 180],   # Pink
    1: [0, 255, 0],       # Green
    2: [128, 0, 128]      # Purple
}

# Directory containing images
image_dir = '/Users/bearcbass/RL4Wound/data/MouseData/train/0'

# Number of clusters
n_clusters = 3

'''
Ok so this is what i have to do. 

1. Set up a function that applys k-means and gets our pixel area
2. Set up the data pipeline and run the k-means/area function on each image
3. record the data and then set it up for the graph.

ok so now that we have a list of images
now lets put it through the thingy and print out the area

Go team go
'''

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

df = pd.read_csv('/Users/bearcbass/RL4Wound/data/MouseData/all_train_imgs.csv')

wound_number = 2
side = 'L'
age = 'Y8'

filtered_df = df[(df['WNum'] == wound_number) & (df['Side'] == side) & (df['Age']== age)]

filtered_df = filtered_df.sort_values(by='Day')

image_names = filtered_df['ImNa'].tolist()

print(image_names)

print(filtered_df)

def list_of_images(df, wound_number, side, age):
    filtered_df = df[(df['WNum'] == wound_number) & (df['Side'] == side) & (df['Age']== age)]

    filtered_df = filtered_df.sort_values(by='Day')

    image_names = filtered_df['ImNa'].tolist()

    return image_names

# we are gonna work with image_names

wound_areas = []
train_dir = '/Users/bearcbass/RL4Wound/data/MouseData/train/'

for image_name in image_names:
    image_found = False

    for sub_dir in range(16):
        image_path = os.path.join(train_dir, str(sub_dir), image_name)

        if os.path.exists(image_path):
            n_clusters = 3
            cluster_colors = [(255,0,0),(0,255,0),(0,0,255)]

            img, clustered_img_array, wound_area = apply_kmeans(image_path, n_clusters, cluster_colors)

            wound_areas.append(wound_area)

            image_found = True
            break
    
    if not image_found:
        print(f"Image {image_name} not found in any subdirectory.")

print(wound_areas)

first_day_area = wound_areas[0]
normalized_wound_areas = wound_areas / first_day_area

"""
This is our graph 
"""
days = list(range(1, len(wound_areas) + 1))  # Example: [1, 2, 3, ..., 16]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(days, normalized_wound_areas, marker='o', linestyle='-', color='b', label='Wound Area')

# Adding labels and title
plt.xlabel('Day')
plt.ylabel('Wound Area (pixels)')
plt.title('Wound Area Over Time')
plt.grid(True)

# Optionally, add a legend
plt.legend()

# Show the plot
plt.show()