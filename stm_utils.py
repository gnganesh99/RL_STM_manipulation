# Classes and functionst to analyze nanonis stm data - .sxm file

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sklearn
from sklearn.linear_model import LinearRegression

# Documentation: https://github.com/hoffmanlabcoding/stmpy/blob/main/stmpy/doc/Stmpy%20101%20-%20getting%20started.ipynb
import stmpy

class Sxm_Image():
    
    def __init__(self, file_name):
        self.file = stmpy.load(str(file_name))
        
        # Metadata of the sxm file
        self.header = self.file.header

        # Size of the scan frame in (m)
        self.frame = self.file.header.get('scan_range')[0]

        # Pixels in an image
        self.pixels = self.file.header.get('scan_pixels')

        self.scan_offset = self.file.header.get('scan_offset')

        self.scan_angle = self.file.header.get('scan_angle')

        self.scan_dir =  self.file.header.get('scan_dir')


    def image(self, channel = "Z_Fwd", linear_correction = True): # If nothing is provided the default channel is "ZFwd"
        
        ''' Converts the sxm file to a 2D image array
            In the absence of argument, default channel is "Z_Fwd"
        '''
        image  = self.file.channels[channel]
        
        # linear baseline correction of the image
        if linear_correction == True:
            image = image_linear_correction(image)

        # correcting the y-coords that are reversed for scan_dir = 'down'
        if self.scan_dir == 'down':
            image = reverse_2D_y(image)

        # if the scan is in backward direction, reverse in x-axis.
        if bkd_scan(channel) == True:
            image = reverse_2D_x(image)

        return image

    def get_channels(self):
        ''' Outputs the channel names in the sxm file
        ''' 
        channel_names = []
        for key in self.file.channels:
            channel_names.append(key)
        return channel_names
    

def bkd_scan(channel_name):
    ch_arr = channel_name.split("_")
    bkd = False
    if ch_arr[-1] == "Bkd":
        bkd = True
    return bkd


def reverse_2D_y(img):
    
    # Reverses the y_axis of an image. useful to correlate labview coords with real coords
    
    img_yr = np.zeros(np.shape(img))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_yr[i, j] = img[(img.shape[0]-1)-i, j]
            
    return img_yr

def reverse_2D_x(img):
    
    # Reverses the y_axis of an image. useful to correlate labview coords with real coords
    
    img_xr = np.zeros(np.shape(img))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_xr[i, j] = img[i, (img.shape[1]-1)-j]
            
    return img_xr


def linear_corrected(y):

    y = np.asarray(y)
    x = np.linspace(1,len(y), len(y))
    X = np.asarray(x).reshape((-1, 1))
    

    reg = LinearRegression(fit_intercept = True).fit(X, y)
    y_corr = y - reg.predict(X)
    
    return y_corr


def image_linear_correction(img):
    img = np.asarray(img)

    im1 = []
    im2 = []

    # Linear correction in the horizontal axis
    for line_ind in range(img.shape[0]):
        line_corr =  linear_corrected(img[line_ind])
        im1.append(line_corr)

    im1 = np.asarray(im1)

    # Linear correction in the vertical axis using transpose
    for line_ind in range(im1.T.shape[0]):
        line_corr =  linear_corrected(im1.T[line_ind])
        im2.append(line_corr)
    
    im2 = np.asarray(im2).T

    return im2