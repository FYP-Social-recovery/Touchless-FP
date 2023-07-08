import cv2
import numpy as np

def display(topic, image, diplay):
    # Display the image
    if diplay:
        cv2.imshow('Image' + topic, image)

# Load the image
img = cv2.imread('M31 - Macro - Without Flash 1.jpg')

display("1", img,True)

# Crop the image to the region of interest (ROI) where the fingerprint is located
roi = img[100:400, 200:500]

display("2", roi,False)

# Convert the ROI to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

display("3", gray,True)

# Define the parameters for the Gabor filter
ksize = 21  # size of the filter kernel
sigma = 4  # standard deviation of the Gaussian envelope
theta = np.pi/4  # orientation of the filter
lambd = 16  # wavelength of the sinusoidal component
gamma = 0.5  # aspect ratio of the filter
psi = 0  # phase offset of the sinusoidal component

# Create the Gabor filter
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

# Apply the Gabor filter to the grayscale image
filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)

# Display the filtered image
display("4",filtered,True)
cv2.waitKey(0)
cv2.destroyAllWindows()





# import cv2
# import numpy as np
# from scipy import ndimage

# img = cv2.imread('input_image.jpg')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ksize = 31  # Size of the filter
# sigma = 5  # Standard deviation of the Gaussian envelope
# theta = 0  # Orientation of the filter (in radians)
# lamda = 10  # Wavelength of the sinusoidal factor
# gamma = 0.5  # Spatial aspect ratio
# psi = 0  # Phase offset

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi)


# kernel /= 1.5 * kernel.sum()

# filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)

# equalized = cv2.equalizeHist(filtered)

# cv2.imshow('Original', gray)
# cv2.imshow('Filtered', filtered)
# cv2.imshow('Equalized', equalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# import cv2
# import numpy as np

# # Load the fingerprint image
# img = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)

# # Preprocess the image
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
# thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# # Segment the fingerprint
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# sure_bg = cv2.dilate(opening, kernel, iterations=3)
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)
# ret, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[unknown == 255] = 0
# markers = cv2.watershed(img, markers)
# img[markers == -1] = [255, 0, 0]

# cv2.imshow('Original', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()