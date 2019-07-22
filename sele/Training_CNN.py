# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 21:45:43 2018

@author: Lin_Shien
"""
import os
import numpy as np
import keras
from SpatialPyramidPooling import SpatialPyramidPooling
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Activation
from keras.layers import BatchNormalization, PReLU, ZeroPadding2D
from keras.utils import np_utils

def get_data():
    images_training = []
    labels_training = []
    
    for i in range(1, 12):
        if(i != 5):
            images_training.append(np.load(r"actHighRail_data" + str(i) + ".npy").astype('float32') / 255)
            label = np.load("actHighRail_labels" + str(i) + ".npy")
            labels_training.append(np_utils.to_categorical(label, 19))
     
    return np.concatenate(images_training), np.concatenate(labels_training)


def get_model():
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
    
    return model


if __name__ == "__main__":
    os.chdir("..\data")
    images_training, labels_training = get_data()
    
    images_testing = np.load("actHighRail_data5.npy").astype('float32') / 255 
    images_testing = np.reshape(images_testing, (3772, 45, 30, 1))
    labels_testing = np.load("actHighRail_labels5.npy")
    labels_testing = np.reshape(labels_testing, (labels_testing.shape[0]))
    labels_testing = np_utils.to_categorical(labels_testing, 19)

    numOfTrainingImages = images_training.shape[0]
    numOfClasses = 19
    batch_size = 200
    epochs = 1

    #rand_state = np.random.RandomState(1337)
    #rand_state.shuffle(images_training)
    #rand_state.seed(1337)
    #rand_state.shuffle(labels_training)

    
    model = get_model()
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
    
    score = model.evaluate(images_testing, labels_testing, verbose = 1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    
