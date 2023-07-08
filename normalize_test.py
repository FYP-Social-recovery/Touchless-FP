
import cv2
import numpy as np

from normalization import normalize
from segmentation import Segmentation
from orientation import calculate_angles
from frequency import ridge_freq
from gabor_filter import gabor_filter
from skeletonize import skeletonize
 
# Reading the image from the present directory
image = cv2.imread("Finger detection - 4.jpg")
# Resizing the image for compatibility
image = cv2.resize(image, (500, 600))

cv2.imshow("original image", image)
 
# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("grey image", image_bw)
cv2.imwrite("1 - grey image.jpg",image_bw)

# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(8,8))
image_bw = clahe.apply(image_bw) + 30  # + 30 makes background black

cv2.imshow("clehe image", image_bw)
cv2.imwrite("1 - clehe image.jpg",image_bw)

block_size = 16
 
# normalization - removes the effects of sensor noise and finger pressure differences.
normalized_img = normalize(image_bw.copy(), float(100), float(100))

cv2.imshow("normalized image", normalized_img)
cv2.imwrite("1 - normalized image.jpg",normalized_img)

# normalisation
(segmented_img, normim, mask) = Segmentation.create_segmented_and_variance_images(normalized_img, block_size, 0.2)

# orientations
angles = calculate_angles(normalized_img, W=block_size, smoth=False)

# find the overall frequency of ridges in Wavelet Domain
freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

# create gabor filter and do the actual filtering
gabor_img = gabor_filter(normim, angles, freq)

cv2.imshow("gabor image", gabor_img)
cv2.imwrite("1 - gabor image.jpg",gabor_img)

# thinning or skeletonize
thin_image = skeletonize(gabor_img)


cv2.imshow("thin image", thin_image)
cv2.imwrite("1 - thin image.jpg",thin_image)

cv2.waitKey(0)
cv2.destroyAllWindows()