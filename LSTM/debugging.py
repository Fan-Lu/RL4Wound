import cv2
import numpy as np

# Load and preprocess the image
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_Y8-2-L.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a mask to ignore black corners
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Apply the mask to the grayscale image
gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = gray_masked[y, x]
        print(f"Pixel value at ({x},{y}): {pixel_value}")

# Display the masked image and set up the mouse callback
cv2.imshow('Masked Image', gray_masked)
cv2.setMouseCallback('Masked Image', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()