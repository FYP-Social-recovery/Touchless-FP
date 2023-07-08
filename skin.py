import cv2
import numpy as np


# # ----------------------- Step 1 (Done) - Identify Background and finger - background black / finger white --------------------------------------


# #Open a simple image
# img=cv2.imread("M31 - Macro - Without Flash 1.jpg")

# #converting from gbr to YCbCr color space
# img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
# #skin color range for YCbCr color space 
# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) # background black / finger white
# YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) #It is useful in removing noise


# cv2.imwrite("Finger detection - 1.jpg",YCrCb_mask)


# # ----------------------- Step 2 (Done) - Identify Background and finger - background white / finger black --------------------------------------


# #Open a simple image
# img=cv2.imread("M31 - Macro - Without Flash 1.jpg")

# #converting from gbr to YCbCr color space
# img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# #skin color range for YCbCr color space 
# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) # background black / finger white
# # Inverting the mask by
# YCrCb_mask = cv2.bitwise_not(YCrCb_mask)  # background white / finger black
# YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) #It is useful in removing noise


# cv2.imwrite("Finger detection - 2.jpg",YCrCb_mask)


# # ----------------------- Step 3 (Done) - Remove Background - background black --------------------------------------


# #Open a simple image
# img=cv2.imread("M31 - Macro - Without Flash 1.jpg")

# #converting from gbr to YCbCr color space
# img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# #skin color range for YCbCr color space 
# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))  # background black / finger white
# YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) #It is useful in removing noise

# YCrCb_result = cv2.bitwise_and(img,img,mask=YCrCb_mask)

# cv2.imwrite("Finger detection - 3.jpg",YCrCb_result)


# # ----------------------- Step 4 (Done) - Remove Background - background white --------------------------------------


#Open a simple image
img=cv2.imread("M31 - Macro - With Flash (1).jpg")

#converting from gbr to YCbCr color space
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#skin color range for YCbCr color space 
YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))  # background black / finger white
YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) #It is useful in removing noise

YCrCb_result = cv2.bitwise_and(img,img,mask=YCrCb_mask)

# get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
black_pixels = np.where(
    (YCrCb_result[:, :, 0] == 0) & 
    (YCrCb_result[:, :, 1] == 0) & 
    (YCrCb_result[:, :, 2] == 0)
)

# set those pixels to white
YCrCb_result[black_pixels] = [255, 255, 255]

cv2.imwrite("1 - YCrCb_result.jpg",YCrCb_result)


# # ----------------------- Step 5 (Done) - Gray scale transformation --------------------------------------

# #Open a simple image
# img=cv2.imread("M31 - Macro - Without Flash 1.jpg")

# #converting from gbr to YCbCr color space
# img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# #skin color range for YCbCr color space 
# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))  # background black / finger white
# YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) #It is useful in removing noise

# YCrCb_result = cv2.bitwise_and(img,img,mask=YCrCb_mask)

# # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
# black_pixels = np.where(
#     (YCrCb_result[:, :, 0] == 0) & 
#     (YCrCb_result[:, :, 1] == 0) & 
#     (YCrCb_result[:, :, 2] == 0)
# )

# # set those pixels to white
# YCrCb_result[black_pixels] = [255, 255, 255]

# gray_scale_image = cv2.cvtColor(YCrCb_result, cv2.COLOR_BGR2GRAY)

# cv2.imwrite("Finger detection - 5.jpg",gray_scale_image)


# # ----------------------- Step 6 (Not done) - Fingerprint image enhancement --------------------------------------


# from matplotlib import pyplot as plt
# import numpy as np
# #Open a simple image
# img=cv2.imread("M31 - Macro - Without Flash 1.jpg")


# # define a function to compute and plot histogram
# def plot_histogram(img, title, mask=None):
#    # split the image into blue, green and red channels
#    channels = cv2.split(img)
#    colors = ("b", "g", "r")
#    plt.title(title)
#    plt.xlabel("Bins")
#    plt.ylabel("# of Pixels")
#    # loop over the image channels
#    for (channel, color) in zip(channels, colors):
#       # compute the histogram for the current channel and plot it
#       hist = cv2.calcHist([channel], [0], mask, [256], [0, 256])
#       plt.plot(hist, color=color)
#       plt.xlim([0, 256])

# # define a mask for our image; black for regions to ignore

# # and white for regions to examine
# mask = np.zeros(img.shape[:2], dtype="uint8")
# cv2.rectangle(mask, (160, 130), (410, 290), 255, -1)

# # display the masked region
# masked = cv2.bitwise_and(img, img, mask=mask)

# # compute a histogram for masked image
# plot_histogram(img, "Histogram for Masked Image", mask=mask)

# # show the plots
# plt.show()
# cv2.imshow("Mask", mask)
# cv2.imshow("Mask Image", masked)
# cv2.waitKey(0)




# ----------------------------------------------------------------


# #Open a simple image
# img=cv2.imread("M31 - Macro - Without Flash 1.jpg")

# #converting from gbr to YCbCr color space
# img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# #skin color range for YCbCr color space 
# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))  # background black / finger white
# YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) #It is useful in removing noise

# YCrCb_result = cv2.bitwise_and(img,img,mask=YCrCb_mask)

# # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
# black_pixels = np.where(
#     (YCrCb_result[:, :, 0] == 0) & 
#     (YCrCb_result[:, :, 1] == 0) & 
#     (YCrCb_result[:, :, 2] == 0)
# )

# # set those pixels to white
# YCrCb_result[black_pixels] = [255, 255, 255]

# gray_scale_image = cv2.cvtColor(YCrCb_result, cv2.COLOR_BGR2GRAY)

# cv2.imwrite("Finger detection - 5.jpg",gray_scale_image)

  
# # Reading the image named 'input.jpg'
# img = cv2.imread("Finger detection - 3.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  

# # Choose a kernel size
# kernel_size = 5


# # Calculate the local mean and standard deviation using a Gaussian blur
# blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
# local_mean = cv2.blur(img, (kernel_size, kernel_size))

# # Calculate the local standard deviation
# local_std = cv2.absdiff(blur, local_mean)
# local_std = cv2.convertScaleAbs(local_std)
# local_std = cv2.sqrt(local_std)

# # Set the threshold
# threshold = 2 * local_std

# # Apply the filter
# filtered = img.copy()
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if abs(img[i,j] - local_mean[i,j]) > threshold[i,j]:
#             filtered[i,j] = local_mean[i,j]

# # Display the filtered image
# cv2.imshow('Filtered Image', filtered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# CLEHE

# img = cv2.imread("M31 - Macro - With Flash.jpg")
# image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img = clahe.apply(img)
# img1 = img * 255
# a = "CLAHE"
# cv2.imshow("Clahe",img)