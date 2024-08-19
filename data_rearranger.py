####################################################
# Description: Data Rearranger for Training/Val purposes
# Version: V0.0.1
# Author: Sebastian Osorio @ UCSC
# Data: 2024-5-17
####################################################
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

"""
Function: get_group_paths
Inputs: 
base_path (string) = directory pathing to either train/val
group_name (string) = which HIPM group we are looking at
groupings (dictonary) = which subdirectories are labeled with HIPM
Outputs: a list of all the subdirectories under the group_name 

HIPM (Hemostasis, Inflammation, Proliferation, Maturation)
"""
def get_group_paths(base_path, group_name, groupings):
    if group_name not in groupings:
        raise ValueError(f"Group name '{group_name}' not found in groupings.")
    return [os.path.join(base_path, sub_dir) for sub_dir in groupings[group_name]]

"""
Function: get_random_image
Inputs:
base_path (string): main directory
subdirs (list): list of subdirectory names
all_images (list): list of all images already accumulated

Output:
str: The path of a randomly selected image.
"""
def get_random_images(base_path, subdirs, num_images):
    # Collect all image paths from the given subdirectories
    all_images = []
    for subdir in subdirs:
        if os.path.isdir(subdir):
            for file_name in os.listdir(subdir):
                file_path = os.path.join(subdir, file_name)
                if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(file_path)
    
    if len(all_images) < num_images:
        raise ValueError(f"Not enough images in the specified subdirectories. Required: {num_images}, Found: {len(all_images)}")
    
    # Randomly select one image
    return random.sample(all_images, num_images)

"""
Function: get_ordered_images
This is supposed to return a list of images that can be used for training
Parameters:
base_path: main directory to data/mouseData/train or val
groupings: dictionary with all the subdirectories 
order: the order the wounds will be compiled in
num_images_list: how many images for each event
Output:
A list of all the ordered images for the setup you have.
"""
def get_ordered_images(base_path, groupings, order, num_images_list):
    if len(order) != len(num_images_list):
        raise ValueError("The length of 'order' and 'num_images_list' must be the same.")
    
    ordered_images = []
    
    for group_name, num_images in zip(order, num_images_list):
        if group_name not in groupings:
            raise ValueError(f"Group name '{group_name}' not found in groupings.")
        
        subdirs = get_group_paths(base_path, group_name, groupings)
        group_images = get_random_images(base_path, subdirs, num_images)
        ordered_images.extend(group_images)
    
    return ordered_images

""""
images stuff
"""

def plot_images(image_paths, target_size=(100, 100)):
    num_images = len(image_paths)
    cols = 4  # Number of columns in the grid
    rows = (num_images + cols - 1) // cols  # Number of rows in the grid

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  # Flatten in case axes is a 2D array

    for ax, img_path, day in zip(axes, image_paths, range(1, num_images + 1)):
        img = Image.open(img_path)
        img = img.resize(target_size, Image.LANCZOS)  # Resize the image with antialiasing
        ax.imshow(img)
        ax.axis('off')  # Hide the axis
        ax.set_title(f'Day {day}', fontsize=10, pad=5)  # Add the day label
        
    # Hide any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Keeps track of pathing from main directory.
path_to_train = "data/MouseData/train"
path_to_val = "data/MouseData/val"

# Create a dictionary to store the grouping
# Adjust accordingly - 
# I picked random values from the heart - Sebastian (May 2024)
groupings = {
    'hemostasis': ['0', '1'],
    'inflammation': ['2', '3',],
    'proliferation': ['4', '5', '6', '7' , '8', '9'],
    'maturation' : ['10', '11', '12', '13', '14', '15']
}

order = ['hemostasis', 'inflammation', 'proliferation', 'maturation']
num_images_list = [2, 14, 0, 0]

# Get the ordered list of images
ordered_images = get_ordered_images(path_to_train, groupings, order, num_images_list)
print("Ordered list of images:", ordered_images)

plot_images(ordered_images)