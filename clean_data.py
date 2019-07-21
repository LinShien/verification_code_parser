# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:55:55 2018

@author: Lin_Shien
"""
from scipy.io import loadmat
import numpy as np
import cv2
import random
from keras.utils import np_utils

def rotateImage(img, angle, center = None, scale = 1.0) :
    image = img.copy()
    (h, w) = image.shape[: 2]
    
    if center is None :
        center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)    #產生2 x 3仿射矩陣
    result = cv2.warpAffine(image, M, (w, h))
    return result.copy()

def create_dirt_bound(img):
    image = img.copy()
    thickness = random.randint(1, 8)     # the thickness of dirt
    place = random.randint(0, 3)         # 0 => left, 1 => up, 2 => right, 3 => down
    

    if place == 0 or place == 2:
        start = 0
        end = 0
        while start == end:
            start = random.randint(0, image.shape[0])
            end = random.randint(0, image.shape[0])
        if start > end:
            temp = start
            start = end
            end = temp
        if place == 0:
            image[start : end + 1, 0 : thickness, :] = 255
        
        if place == 2:
            image[start : end + 1, image.shape[1] - thickness : image.shape[1] + 1, :] = 255
    
    if place == 1 or place == 3:
        start = 0
        end = 0
        while start == end:
            start = random.randint(0, image.shape[1])
            end = random.randint(0, image.shape[1])
        if start > end:
            temp = start
            start = end
            end = temp
        if place == 1:
            image[0 : thickness, start : end + 1, :] = 255
        
        if place == 3:
            image[image.shape[0] - thickness : image.shape[0] + 1, start : end + 1, :] = 255
    
    return image

def create_dirt(img):
    image = img.copy()
    
    numOfDirt = random.randint(10, 20)
    
    for i in range(numOfDirt):
        x = 10000
        y = 10000
        
        while(x + 1 > image.shape[1] or y + 1 > image.shape[0]):
            x = random.randint(0, image.shape[1])
            y = random.randint(0, image.shape[0])
        image[y : y + 2, x : x + 2, :] = 255
    
    return image.copy()
    
'''
data = loadmat("emnist-balanced")
numOfTrainingImages = data['dataset'][0][0][0][0][0][0].shape[0]
numOfTestingImages = data['dataset'][0][0][1][0][0][0].shape[0]
numOfClasses = 47

data_train = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255
data_clone = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255
data_clone2 = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255
data_clone3 = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255
data_clone4 = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255
data_clone5 = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255
data_clone6 = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255
data_clone7 = np.reshape(data['dataset'][0][0][0][0][0][0][: numOfTrainingImages], (numOfTrainingImages, 28, 28, 1)) / 255

la1 = np_utils.to_categorical(data['dataset'][0][0][0][0][0][1][:numOfTrainingImages], numOfClasses)

data_test = np.reshape(data['dataset'][0][0][1][0][0][0][: numOfTestingImages], (numOfTestingImages, 28, 28, 1)) / 255
la2 = np_utils.to_categorical(data['dataset'][0][0][1][0][0][1][: numOfTestingImages], numOfClasses)

for i in range(numOfTrainingImages) :            # transpose image first, then rotate it with 90 degrees counterclockwise
    data_train[i] = np.rot90(np.fliplr(data_train[i]))

for i in range(numOfTestingImages) :
    angel = random.randint(0, 60)
    data_clone[i] = np.reshape(rotateImage(data_train[i], angel), (28, 28, 1))    # randomly rotate a image
    
    angel = random.randint(0, 60)
    data_clone2[i] = np.reshape(rotateImage(data_train[i], angel), (28, 28, 1)) 
    
    angel = random.randint(0, 60)
    data_clone3[i] = np.reshape(rotateImage(data_train[i], angel), (28, 28, 1)) 
    
    angel = random.randint(-60, 0)
    data_clone4[i] = np.reshape(rotateImage(data_train[i], angel), (28, 28, 1)) 
    
    angel = random.randint(-60, 0)
    data_clone5[i] = np.reshape(rotateImage(data_train[i], angel), (28, 28, 1)) 
    
    angel = random.randint(0, 60)
    data_clone6[i] = np.reshape(rotateImage(data_train[i], angel), (28, 28, 1))    # randomly rotate a image
    
    angel = random.randint(-60, 0)
    data_clone7[i] = np.reshape(rotateImage(data_train[i], angel), (28, 28, 1)) 
    
np.save("data_train_Bal2.npy", np.concatenate((data_train, data_clone, data_clone2, data_clone4, data_clone5, data_clone6, data_clone7)))
np.save("data_test_Bal2.npy", np.concatenate((data_test, data_clone3)))
np.save("train_label_Bal2.npy", np.concatenate((la1, la1, la1, la1, la1, la1, la1)))
np.save("test_label_Bal2.npy", np.concatenate((la2, la1)))
'''

data = np.load('Arial_data.npy')
labels = np.load('Arial_labels.npy')
data = data[:, :, :, 0 : 1]
la1 = np_utils.to_categorical(labels, 36)

data_clone1 = data.copy()
data_clone2 = data.copy()

data_clone3 = data.copy()
'''
data_clone4 = data.copy()
data_clone5 = data.copy()
data_clone6 = data.copy()
data_clone7 = data.copy()
data_clone8 = data.copy()
data_clone9 = data.copy()
data_clone10 = data.copy()
data_clone11 = data.copy()
data_clone12 = data.copy()
data_clone13 = data.copy()
data_clone14 = data.copy()
data_clone15 = data.copy()
data_clone16 = data.copy()
data_clone17 = data.copy()
data_clone18 = data.copy()
data_clone19 = data.copy()
data_clone20 = data.copy()
'''

for i in range(data.shape[0]):
    angel = random.randint(0, 20)
    data_clone1[i] = np.reshape(rotateImage(data[i], angel), (30, 40, 1))
    
    #angel = random.randint(0, 20)
    #data_clone2[i] = np.reshape(rotateImage(data[i], angel), (30, 40, 1))
    
    data_clone2[i] = create_dirt(np.reshape(data[i], (30, 40, 1)))
    
    data_clone3[i] = create_dirt_bound(np.reshape(data[i], (30, 40, 1)))
    
    '''
    data_clone5[i] = create_dirt_bound(np.reshape(data[i], (30, 40, 1)))
    data_clone6[i] = create_dirt_bound(np.reshape(data[i], (30, 40, 1)))
    
      
    angel = random.randint(0, 45)
    data_clone7[i] = create_dirt_bound(np.reshape(rotateImage(data[i], angel), (30, 40, 1)))
    
    angel = random.randint(0, 45)
    data_clone8[i] = create_dirt_bound(np.reshape(rotateImage(data[i], angel), (30, 40, 1)))
    
    angel = random.randint(0, 45)
    data_clone9[i] = create_dirt_bound(np.reshape(rotateImage(data[i], angel), (30, 40, 1)))

    data_clone10[i] = create_dirt(np.reshape(data[i], (30, 40, 1))).astype('float32') / 255
    data_clone11[i] = create_dirt(np.reshape(data[i], (30, 40, 1))).astype('float32') / 255
    data_clone12[i] = create_dirt(np.reshape(data[i], (30, 40, 1))).astype('float32') / 255

    angel = random.randint(0, 45)
    data_clone13[i] = create_dirt(np.reshape(rotateImage(data[i], angel), (30, 40, 1)))
    
    angel = random.randint(0, 45)
    data_clone14[i] = create_dirt(np.reshape(rotateImage(data[i], angel), (30, 40, 1)))
    
    angel = random.randint(0, 45)
    data_clone15[i] = create_dirt(np.reshape(rotateImage(data[i], angel), (30, 40, 1)))

    data_clone16[i] = create_dirt_bound(create_dirt(np.reshape(data[i], (30, 40, 1))))
    data_clone17[i] = create_dirt_bound(create_dirt(np.reshape(data[i], (30, 40, 1))))
    data_clone18[i] = create_dirt_bound(create_dirt(np.reshape(data[i], (30, 40, 1))))

    angel = random.randint(0, 45)
    data_clone19[i] = create_dirt_bound(create_dirt(np.reshape(rotateImage(data[i], angel), (30, 40, 1))))
    
    angel = random.randint(0, 45)
    data_clone20[i] = create_dirt_bound(create_dirt(np.reshape(rotateImage(data[i], angel), (30, 40, 1))))
    '''
data = data.astype('float32') / 255
data_clone1 = data_clone1.astype('float32') / 255
data_clone2 = data_clone2.astype('float32') / 255
data_clone3 = data_clone3.astype('float32') / 255
'''
data_clone4 = data_clone4.astype('float32') / 255
data_clone5 = data_clone5.astype('float32') / 255
data_clone6 = data_clone6.astype('float32') / 255
data_clone7 = data_clone7.astype('float32') / 255
data_clone8 = data_clone8.astype('float32') / 255
data_clone9 = data_clone9.astype('float32') / 255
data_clone10 = data_clone10.astype('float32') / 255
data_clone11 = data_clone11.astype('float32') / 255
data_clone12 = data_clone12.astype('float32') / 255
data_clone13 = data_clone13.astype('float32') / 255
data_clone14 = data_clone14.astype('float32') / 255
data_clone15 = data_clone15.astype('float32') / 255
data_clone16 = data_clone16.astype('float32') / 255
data_clone17 = data_clone17.astype('float32') / 255
data_clone18 = data_clone18.astype('float32') / 255
data_clone19 = data_clone19.astype('float32') / 255
data_clone20 = data_clone20.astype('float32') / 255 
'''
np.save("Arial_data_train_140k.npy", np.concatenate((data, data_clone1, data_clone2, data_clone3)))

np.save("Arial_labels_140k.npy", np.concatenate((la1, la1, la1, la1)))
'''
dataTrain = list()
dataTest = list()

for i in range(100, 300):
    newimg = eliminateCurve(r'C:\Users\Lin_Shien\Desktop\project\train_img' + '\image' + str(i) + '.jpg')
    p1, p2, p3, p4 = deleCurve_and_create_pieces(newimg)
    dataTrain.append(cv2.resize(p1, dsize=(45, 25), interpolation=cv2.INTER_CUBIC))
    dataTrain.append(cv2.resize(p2, dsize=(45, 25), interpolation=cv2.INTER_CUBIC))
    dataTrain.append(cv2.resize(p3, dsize=(45, 25), interpolation=cv2.INTER_CUBIC))
    dataTrain.append(cv2.resize(p4, dsize=(45, 25), interpolation=cv2.INTER_CUBIC))
for i in range(0, 25):
    newimg = eliminateCurve(r'C:\Users\Lin_Shien\Desktop\project\test_img'+ '\image' + str(i) + '.jpg')
    p1, p2, p3, p4 = deleCurve_and_create_pieces(newimg)
    dataTest.append(cv2.resize(p1, dsize=(45, 25), interpolation=cv2.INTER_CUBIC))
    dataTest.append(cv2.resize(p2, dsize=(45, 25), interpolation=cv2.INTER_CUBIC))
    dataTest.append(cv2.resize(p3, dsize=(45, 25), interpolation=cv2.INTER_CUBIC))
    dataTest.append(cv2.resize(p4, dsize=(45, 25), interpolation=cv2.INTER_CUBIC))   
dataTrain = np.array(dataTrain)
dataTest = np.array(dataTest)
'''