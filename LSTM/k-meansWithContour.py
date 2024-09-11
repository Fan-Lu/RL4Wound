import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_Y8-2-L.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define different thresholds for testing
thresholds = [(50, 150), (100, 200), (150, 250)]

# Set up the plot
plt.figure(figsize=(12, 8))

for i, (low, high) in enumerate(thresholds):
    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, low, high)
    
    # Display the results
    plt.subplot(1, len(thresholds), i + 1)
    plt.title(f'Thresholds: {low}, {high}')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

plt.show()
