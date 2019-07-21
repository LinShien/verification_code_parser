# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:29:59 2018

@author: Lin_Shien
"""

from scipy.io import loadmat
import numpy as np
import cv2
import random
from keras.utils import np_utils

def rotateImage(image, angle, center = None, scale = 1.0) :
    (h, w) = image.shape[: 2]
    
    if center is None :
        center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)    #產生2 x 3仿射矩陣
    result = cv2.warpAffine(image, M, (w, h))
    return result

data = loadmat("emnist-balanced")
numOfTrainingImages = data['dataset'][0][0][0][0][0][0].shape[0]
numOfTestingImages = data['dataset'][0][0][1][0][0][0].shape[0]
numOfClasses = 47

data_train = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255
la = data['dataset'][0][0][0][0][0][1][: numOfTrainingImages]


# transpose image first, then rotate it with 90 degrees counterclockwise
for i in range(numOfTrainingImages) :                    
    data_train[i] = np.rot90(np.fliplr(data_train[i]))
    
angel = random.randint(-45, 45)
dig1 = np.reshape(rotateImage(data_train[100], angel), (28, 28, 1))

angel = random.randint(-45, 45)
dig2 = np.reshape(rotateImage(data_train[101], angel), (28, 28, 1))

angel = random.randint(-45, 45)
dig3 = np.reshape(rotateImage(data_train[102], angel), (28, 28, 1))

angel = random.randint(-45, 45)
dig4 = np.reshape(rotateImage(data_train[103], angel), (28, 28, 1))

angel = random.randint(-45, 45)
four_digits_image = np.concatenate((data_train[100], data_train[101], data_train[102], data_train[103]), axis = 1)

cv2.imwrite("4digs6.jpg", four_digits_image * 255)