import os
import matplotlib.pyplot as plt
import cv2

# List of image paths
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

# Number of images
n_images = len(image_paths)

# Create a grid for displaying the images
plt.figure(figsize=(15, 15))

# Loop through the image paths and display each image in the grid
for i, image_path in enumerate(image_paths):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV) to RGB (matplotlib)
    
    # Add each image to the plot
    plt.subplot(4, 4, i + 1)  # Adjust the grid size (4x4 in this case for 16 images)
    plt.imshow(img)
    plt.title(f'Day {i}')
    plt.axis('off')  # Turn off axis

# Display the grid of images
plt.tight_layout()
plt.show()