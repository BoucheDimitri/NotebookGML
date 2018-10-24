from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
path=os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *
from spectral_clustering.func import *


def image_segmentation(input_img):
    #      a skeleton function to perform image segmentation, needs to be completed
    #  Input
    #  input_img:
    #      (string) name of the image file, without extension (e.g. 'four_elements.bmp')
    filename = path+'/'+input_img

    X = io.imread(filename)
    X=(X - np.min(X)) / (np.max(X) - np.min(X))

    im_side = np.size(X,1)
    Xr=X.reshape(im_side**2,3)
    #################################################################
    # Y_rec should contain an index from 1 to c where c is the      #
    # number of segments you want to split the image into           #
    #################################################################

    var=

    W=
    L = 
    Y_rec = 

    #################################################################
    #################################################################

    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(X)


    plt.subplot(1,2,2)
    Y_rec=Y_rec.reshape(im_side,im_side)
    plt.imshow(Y_rec)
    
    plt.show()

