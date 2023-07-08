
import cv2
import numpy as np
 
# Reading the image from the present directory
image = cv2.imread("Finger detection - 5.jpg")
# Resizing the image for compatibility
image = cv2.resize(image, (500, 600))
 
# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(8,8))
final_img = clahe.apply(image_bw) + 30  # + 30 makes background black


cv2.imshow("CLAHE image", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()