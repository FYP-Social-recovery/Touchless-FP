import cv2
import numpy as np

# Reading the image named 'input.jpg'
img = cv2.imread("Finger detection - 3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  

# Choose a kernel size
kernel_size = 5


# Calculate the local mean and standard deviation using a Gaussian blur
blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
local_mean = cv2.blur(img, (kernel_size, kernel_size))

# Calculate the local standard deviation
local_std = cv2.absdiff(blur, local_mean)
local_std = cv2.convertScaleAbs(local_std)
local_std = cv2.sqrt(local_std)

# Set the threshold
threshold = 2 * local_std

# Apply the filter
filtered = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if abs(img[i,j] - local_mean[i,j]) > threshold[i,j]:
            filtered[i,j] = local_mean[i,j]

# Display the filtered image
cv2.imshow('Filtered Image', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()