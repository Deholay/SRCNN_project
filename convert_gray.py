import cv2

# Load the color image
image = cv2.imread('image.png')  # Replace with your image path

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('gray_image.jpg', gray_image)

# Optional: Display the result
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()