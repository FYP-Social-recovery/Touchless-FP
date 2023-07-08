# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 03:15:03 2016

@author: utkarsh
"""


# RIDGEFILTER - enhances fingerprint image via oriented filters
#
# Function to enhance fingerprint image via oriented filters
#
# Usage:
#  newim =  ridgefilter(im, orientim, freqim, kx, ky, showfilter)
#
# Arguments:
#         im       - Image to be processed.
#         orientim - Ridge orientation image, obtained from RIDGEORIENT.
#         freqim   - Ridge frequency image, obtained from RIDGEFREQ.
#         kx, ky   - Scale factors specifying the filter sigma relative
#                    to the wavelength of the filter.  This is done so
#                    that the shapes of the filters are invariant to the
#                    scale.  kx controls the sigma in the x direction
#                    which is along the filter, and hence controls the
#                    bandwidth of the filter.  ky controls the sigma
#                    across the filter and hence controls the
#                    orientational selectivity of the filter. A value of
#                    0.5 for both kx and ky is a good starting point.
#         showfilter - An optional flag 0/1.  When set an image of the
#                      largest scale filter is displayed for inspection.
# 
# Returns:
#         newim    - The enhanced image
#
# See also: RIDGEORIENT, RIDGEFREQ, RIDGESEGMENT

# Reference: 
# Hong, L., Wan, Y., and Jain, A. K. Fingerprint image enhancement:
# Algorithm and performance evaluation. IEEE Transactions on Pattern
# Analysis and Machine Intelligence 20, 8 (1998), 777 789.

### REFERENCES

# Peter Kovesi  
# School of Computer Science & Software Engineering
# The University of Western Australia
# pk at csse uwa edu au
# http://www.csse.uwa.edu.au/~pk



import numpy as np
import scipy
def ridge_filter(im, orient, freq, kx=0.65, ky=0.65):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    return_img = np.zeros((rows,cols))

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    
    freq_1d = freq.flatten()
    frequency_ind = np.array(np.where(freq_1d>0))
    non_zero_elems_in_freq = freq_1d[frequency_ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq*100)))/100
    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.
    sigma_x = 1/unfreq*kx
    sigma_y = 1/unfreq*ky
    print(sigma_x,sigma_y)
    block_size = int(3*np.max([sigma_x,sigma_y]))
    array = np.linspace(-block_size,block_size,(2*block_size + 1))
    x, y = np.meshgrid(array, array)

    # gabor filter equation
    reffilter = np.exp(-(((np.power(x,2))/(sigma_x*sigma_x) + (np.power(y,2))/(sigma_y*sigma_y)))) * np.cos(2*np.pi*unfreq[0]*x)
    filt_rows, filt_cols = reffilter.shape
    gabor_filter = np.array(np.zeros((180//angleInc, filt_rows, filt_cols)))

    # Generate rotated versions of the filter.
    for degree in range(0,180//angleInc):
        rot_filt = scipy.ndimage.rotate(reffilter,-(degree*angleInc + 90),reshape = False)
        gabor_filter[degree] = rot_filt

    # Convert orientation matrix values from radians to an index value that corresponds to round(degrees/angleInc)
    maxorientindex = np.round(180/angleInc)
    orientindex = np.round(orient/np.pi*180/angleInc)
    for i in range(0,rows//16):
        for j in range(0,cols//16):
            if(orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if(orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex

    # Find indices of matrix points greater than maxsze from the image boundary
    block_size = int(block_size)
    valid_row, valid_col = np.where(freq>0)
    finalind = \
        np.where((valid_row>block_size) & (valid_row<rows - block_size) & (valid_col>block_size) & (valid_col<cols - block_size))

    for k in range(0, np.shape(finalind)[1]):
        r = valid_row[finalind[0][k]]; c = valid_col[finalind[0][k]]
        img_block = im[r-block_size:r+block_size + 1][:,c-block_size:c+block_size + 1]
        return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r//16][c//16]) - 1])

    gabor_img = 255 - np.array((return_img < 0)*255).astype(np.uint8)

    return gabor_img  