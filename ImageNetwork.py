#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:04:10 2018

@author: keshavbachu
"""

import keras

def quickModel(inputs, output, possibleOutcomes):
    
    oneHotEncoding = keras.utils.np_utils.to_categorical(output)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(4, kernel_size=3, activation= 'relu', input_shape=(500,500,3)))
    model.add(keras.layers.Conv2D(2, kernel_size=3, activation= 'relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(possibleOutcomes))
    model.add(keras.layers.Activation("softmax"))
    
    #model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'loss'])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(inputs, oneHotEncoding, epochs=3)
    
    temp = model.predict(inputs)
    return temp