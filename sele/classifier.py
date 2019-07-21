# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 21:06:07 2018

@author: Lin_Shien
"""

import numpy as np
import keras
import cv2
from eliminate_curve import eliminateCurve, deleCurve_and_create_pieces
from keras.models import Sequential, load_model
from SpatialPyramidPooling import SpatialPyramidPooling
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.layers import BatchNormalization, Activation, PReLU, ZeroPadding2D

numOfClasses = 47

class LengthMatchException(Exception):                              # 繼承Exception
    pass

def sort(coordinate_list, coord_x_list) :                               # sort the all coordinate based on their area
        for i in range(0, len(coordinate_list) - 1) :
            for j in range(i + 1, len(coordinate_list)) :
                if (coordinate_list[i][2] * coordinate_list[i][3] < coordinate_list[j][2] * coordinate_list[j][3]) :
                    temp1 = coordinate_list[i]
                    coordinate_list[i] = coordinate_list[j]
                    coordinate_list[j] = temp1
                    temp2 = coord_x_list[i]
                    coord_x_list[i] = coord_x_list[j]
                    coord_x_list[j] = temp2
               
def readImageAndCreateRect(img_array):
    #images_detect = cv2.imread(image_name)   # 0 ~ 255
    clone_unchanged = img_array.copy()                                         # 用來locate的，不能修改或在上面標輪廓和長方形

    #gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)                                            # 轉成灰階圖
    ret, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)                           # 轉成二值圖
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # 畫出輪廓
    
    clone = img_array.copy()                                                   # mutable
    cv2.drawContours(clone, contours[: len(contours)], -1, (0, 255, 0), 0)     # clone會被原地修改
    #cv2.imwrite("4con2.jpg", clone)
    
    coord_list = list()
    coord_x_list = list()                              # 用 x座標來記住字母順序

    for cnt in contours[: len(contours)] :
        (x, y, w, h) = cv2.boundingRect(cnt)
        coord_list.append((x, y, w, h))
        coord_x_list.append(x)
        
    return coord_list, coord_x_list, clone_unchanged

def sortAllRect(coord_list, coord_x_list, image_array, numOfNeed):
    sort(coord_list, coord_x_list)                              # sort the all coordinates, and num_list as well 
    
    if numOfNeed > len(coord_list):                         # 長度不match就拋錯
        raise LengthMatchException(numOfNeed)
        
    if numOfNeed != len(coord_list):        
        coord_x_list = coord_x_list[0 : numOfNeed]                  # remove the other rectangulars not needed
        coord_list = coord_list[0 : numOfNeed]
    
    rectNeeded_list = list()
    
    for i in range(numOfNeed):
        max_element = max(coord_x_list)
        index = coord_x_list.index(max_element)
        coord_x_list.remove(max_element)   
        (x, y, w, h) = coord_list[index]
        coord_list.remove(coord_list[index])
        #cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 0)                                         # image_array會被原地修改，長方線會影響辨識率
        rectNeeded_list.insert(i, image_array[y - 3 : y + h + 3, x - 3 : x + w + 3 ,]) 

        #rectNeeded_list[i] = cv2.resize(rectNeeded_list[i], dsize = (28, 28), interpolation=cv2.INTER_CUBIC)       # distortion may happen here
        # 28x28x3 now
    return rectNeeded_list          # return list of rects
    
    
def classify_passcode(img):
    #if rect_list == None or len(rect_list) == 0:
    #    raise TypeError()
        
    numOfClasses = 19   
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1),input_shape = (None, None, 1)))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())                         #BatchNorm    
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    model.add(ZeroPadding2D((1,1)))
        

    model.add(Conv2D(32, (3, 3)))   
    model.add(BatchNormalization())                         #BatchNorm    
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    model.add(MaxPooling2D(pool_size=(2, 2)))               
    model.add(ZeroPadding2D((1,1)))
       
    
    model.add(Conv2D(64, (3, 3)))   
    model.add(BatchNormalization())                         #BatchNorm    
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    model.add(ZeroPadding2D((1,1)))
       

    model.add(Conv2D(64, (3, 3)))   
    model.add(BatchNormalization())                         #BatchNorm    
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    model.add(MaxPooling2D(pool_size=(2, 2)))           
       
    
    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Dense(numOfClasses, activation='softmax'))     #5
    
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    
    inputfile = 'ActHighRail_ep60_32k_Dp'
    model.load_weights(inputfile)
    
    try:
      newimg = eliminateCurve(img, 40)
      p1, p2, p3, p4 = deleCurve_and_create_pieces(newimg)
    except (ValueError, IndexError) as ex:
      try:
        newimg = eliminateCurve(img, 45)
        p1, p2, p3, p4 = deleCurve_and_create_pieces(newimg)
      except (ValueError, IndexError) as ex:  
        newimg = eliminateCurve(img, 50)
        p1, p2, p3, p4 = deleCurve_and_create_pieces(newimg)
      
    #data_to_test = [rect[:, :,] for rect in rect_list]
    
    cv2.resize(p1, dsize = (30, 45), interpolation=cv2.INTER_CUBIC)
    cv2.resize(p2, dsize = (30, 45), interpolation=cv2.INTER_CUBIC)
    cv2.resize(p3, dsize = (30, 45), interpolation=cv2.INTER_CUBIC)
    cv2.resize(p4, dsize = (30, 45), interpolation=cv2.INTER_CUBIC)
    mapping_list = ['2', '3', '4', '5', '7', '9', 'A', 'C', 'F', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'T', 'Y', 'Z']
    
    str_predicted = ""
    '''
    for i in range(len(data_to_test)):                                         # / 255很重要
        score = model.predict(np.reshape(data_to_test[i], (1, data_to_test[i].shape[0], data_to_test[i].shape[1], 1)) / 255, verbose = 1)
        str_mapped = mapping_list[np.argmax(score[0])]  
        str_predicted = str_predicted + str_mapped
    '''
    score = model.predict(np.reshape(p1, (1, p1.shape[0], p1.shape[1], 1)) / 255, verbose = 1)
    str_mapped = mapping_list[np.argmax(score[0])]  
    str_predicted = str_predicted + str_mapped
    
    score = model.predict(np.reshape(p2, (1, p2.shape[0], p2.shape[1], 1)) / 255, verbose = 1)
    str_mapped = mapping_list[np.argmax(score[0])]  
    str_predicted = str_predicted + str_mapped
    
    score = model.predict(np.reshape(p3, (1, p3.shape[0], p3.shape[1], 1)) / 255, verbose = 1)
    str_mapped = mapping_list[np.argmax(score[0])]  
    str_predicted = str_predicted + str_mapped
    
    score = model.predict(np.reshape(p4, (1, p4.shape[0], p4.shape[1], 1)) / 255, verbose = 1)
    str_mapped = mapping_list[np.argmax(score[0])]  
    str_predicted = str_predicted + str_mapped
    
    print(str_predicted)
    return str_predicted

      
#img_array = eliminateCurve('pass_code4.png')
#coord_list, num_list, img_array = readImageAndCreateRect(img_array)
#rect_list = sortAllRect(coord_list, num_list, img_array, 4)
#ans = classify_passcode(rect_list)


