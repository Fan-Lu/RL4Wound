import cv2

# Read the image
image_path = '/Users/bearcbass/RL4Wound/data/MouseData/train/0/Day 0_A8-1-R.png'
image = cv2.imread(image_path, 0)
# Apply bilateral filter
normalized_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Save or display the result
cv2.imwrite('normalized_image.png', normalized_image)
cv2.imshow('Normalized Image', normalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()