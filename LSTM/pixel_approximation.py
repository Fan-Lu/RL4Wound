import cv2
import numpy as np

# Load and preprocess the image
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-1-R.png'
image = cv2.imread(image_path)

cv2.imshow('Wound Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a mask to ignore black corners
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

cv2.imshow('Mask image', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

# Apply Gaussian blur to remove noise
blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)

# Manually threshold the wound pixels (range 60-80 based on your observation)
# This filters only the wound region based on pixel intensity
wound_range = cv2.inRange(blurred, 67, 90)

# Find contours in the thresholded image
contours, _ = cv2.findContours(wound_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out very small contours (that might be noise)
min_contour_area = 100  # Adjust this value as needed
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Ensure there are contours and select the largest one
if len(filtered_contours) > 0:
    largest_contour = max(filtered_contours, key=cv2.contourArea)

    # Calculate the wound area in pixels
    wound_area = cv2.contourArea(largest_contour)
    print(f"Wound pixel area: {wound_area} pixels")

    # Optionally draw the contour for visualization
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    cv2.imshow('Wound Contour', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No valid contours found.")