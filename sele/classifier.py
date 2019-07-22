# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 21:06:07 2018

@author: Lin_Shien
"""

import numpy as np
import keras
import cv2
import Training_CNN
from eliminate_curve import eliminateCurve, deleCurve_and_create_pieces
from keras.models import Sequential
from SpatialPyramidPooling import SpatialPyramidPooling
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.layers import BatchNormalization, Activation, PReLU, ZeroPadding2D

class LengthMatchException(Exception):                                  # 繼承Exception
    pass
     

def classify_verification_code():      
    model = get_model()
    
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    
    inputfile = 'ActHighRail_ep60_32k_Dp'
    model.load_weights(inputfile)
    
    newimg = eliminateCurve('test_img.png', 40)
    deleCurve_and_create_pieces(newimg)
    
    mapping_list = ['2', '3', '4', '5', '7', '9', 'A', 'C', 'F', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'T', 'Y', 'Z']
    
    str_predicted = ""
        
    for i in range(1, 5):
        piece = cv2.imread("p" + str(i) + ".png")
        score = model.predict(np.reshape(piece[:, :, 0 : 1], (1, piece.shape[0], piece.shape[1], 1)) / 255, verbose = 1)
        str_mapped = mapping_list[np.argmax(score[0])]
        str_predicted = str_predicted + str_mapped
    
    print(str_predicted)
    return str_predicted
    

if __name__ == "__main__":        
    classify_verification_code()   


