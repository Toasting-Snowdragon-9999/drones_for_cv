import cv2

# Read image
img = cv2.imread("../images/capture_savannah/img_8.jpg")

# Number of pixels to crop from each side
crop_pixels = 100

# Crop: [y_start:y_end, x_start:x_end]
cropped = img[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]

# Resize image
cropped = cv2.resize(cropped, (1500, 1000))

# Save image
cv2.imwrite("../images/capture_savannah/img_8_cropped.jpg", cropped)

# Show result
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
