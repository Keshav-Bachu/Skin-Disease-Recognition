#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:04:10 2018

@author: keshavbachu
"""

import keras

def quickModel(inputs, output, possibleOutcomes):
    
    model = keras.Sequential()
    model.add(Conv2D(64, kernel_size=3, activation= 'relu', input_shape=(500,500,3)))
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation= 'relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Conv2D(possibleOutcomes, activation= 'spftmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'loss'])
    model.fit(inputs, output, epochs=3)
    
    return model.predict(inputs)