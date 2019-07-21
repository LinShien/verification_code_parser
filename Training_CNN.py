# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 21:45:43 2018

@author: Lin_Shien
"""

import numpy as np
import keras
from SpatialPyramidPooling import SpatialPyramidPooling
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Activation
from keras.layers import BatchNormalization, PReLU, ZeroPadding2D
from keras.utils import np_utils
from eliminate_curve import eliminateCurve, deleCurve_and_create_pieces
'''
dataTrain = np.load('simiToHighRail_data4.npy').astype('float32') / 255 
dataTrain = np.reshape(dataTrain, (480000, 45, 30, 1)) 
dataLabel = np.load('simiToHighRail_label4.npy')   
dataLabel = np_utils.to_categorical(dataLabel, 19)


images_training = np.load("Adobe_data_highRail4.npy")
labels_training = np.load("Adobe_labels_highRail4.npy")

images_training = images_training[:, :, :, 0 : 1].astype('float32') / 255
images_training = np.reshape(images_training, (98800, 45, 30, 1)) 
labels_training = np_utils.to_categorical(labels_training, 19)
'''
d1 = np.load("actHighRail_data1.npy").astype('float32') / 255 
l1 = np.load("actHighRail_labels1.npy")
l1 = np_utils.to_categorical(l1, 19)

d2 = np.load("actHighRail_data2.npy").astype('float32') / 255 
l2 = np.load("actHighRail_labels2.npy")
l2 = np_utils.to_categorical(l2, 19)

d3 = np.load("actHighRail_data3.npy").astype('float32') / 255 
l3 = np.load("actHighRail_labels3.npy")
l3 = np_utils.to_categorical(l3, 19)

d4 = np.load("actHighRail_data4.npy").astype('float32') / 255 
l4 = np.load("actHighRail_labels4.npy")
l4 = np_utils.to_categorical(l4, 19)

d6 = np.load("actHighRail_data6.npy").astype('float32') / 255 
l6 = np.load("actHighRail_labels6.npy")
l6 = np_utils.to_categorical(l6, 19)

d7 = np.load("actHighRail_data7.npy").astype('float32') / 255 
l7 = np.load("actHighRail_labels7.npy")
l7 = np_utils.to_categorical(l7, 19)

d8 = np.load("actHighRail_data8.npy").astype('float32') / 255 
l8 = np.load("actHighRail_labels8.npy")
l8 = np_utils.to_categorical(l8, 19)

d9 = np.load("actHighRail_data9.npy").astype('float32') / 255 
l9 = np.load("actHighRail_labels9.npy")
l9 = np_utils.to_categorical(l9, 19)

d10 = np.load("actHighRail_data10.npy").astype('float32') / 255 
l10 = np.load("actHighRail_labels10.npy")
l10 = np_utils.to_categorical(l10, 19)

d11 = np.load("actHighRail_data11.npy").astype('float32') / 255 
l11 = np.load("actHighRail_labels11.npy")
l11 = np_utils.to_categorical(l11, 19)


d5 = np.load("HihgRail_data.npy").astype('float32') / 255 
d5 = np.reshape(d5, (3772, 45, 30, 1))
l5 = np.load("HihgRail_labels.npy")
l5 = np.reshape(l5, (l5.shape[0]))
l5 = np_utils.to_categorical(l5, 19)

images_training = np.concatenate((d1, d2, d3, d4, d5))
labels_training = np.concatenate((l1, l2, l3, l4, l5))


numOfTrainingImages = images_training.shape[0]
numOfClasses = 19
batch_size = 200
epochs = 1

rand_state = np.random.RandomState(1337)
rand_state.shuffle(images_training)
rand_state.seed(1337)
rand_state.shuffle(labels_training)


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

model.fit(images_training[:], labels_training[:],
         batch_size = batch_size,
         epochs = epochs,
         verbose = 1,
         validation_data = (images_training, labels_training))

score = model.evaluate(images_training[:], labels_training[:], verbose = 1)
print('ideal loss:', score[0])
print('ideal accuracy:', score[1])

score = model.evaluate(d11, l11, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#c.save_weights('PK')
