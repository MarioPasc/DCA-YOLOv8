import cv2

# Load an image
image = cv2.imread('14_002_5_0027.jpeg')

# Check if image is loaded fine
if image is None:
    print("Error opening image! Please check the path.")
    exit()

# Convert the image to YUV format (as CLAHE works on Y channel of YUV image)
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Split the image into Y, U, and V channels
y, u, v = cv2.split(yuv)

# Create a CLAHE object (Arguments are optional)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE to the Y channel
cl_y = clahe.apply(y)

# Merge the channels back together
merged = cv2.merge((cl_y, u, v))

# Convert the image back to BGR format
final_image = cv2.cvtColor(merged, cv2.COLOR_YUV2BGR)

# Save the resulting image
cv2.imwrite('clahe.jpg', final_image)

# Display the original and the CLAHE image
cv2.imshow('Original Image', image)
cv2.imshow('CLAHE Image', final_image)
