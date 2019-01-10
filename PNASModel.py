#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:08:15 2019

@author: keshavbachu
"""

import keras
import tensorflow as tf

#Not truely thePNAS model as I ddint want to make it as big due to hardware limitations
#Modified to what block structure I can hopefully cut out
def trainPNAS(trainInput, trainOutput, numClasses):    
    HC2min = keras.models.Input((trainInput.shape[1], trainInput.shape[2], trainInput.shape[3]))
    HC1min = keras.models.Input((trainInput.shape[1], trainInput.shape[2], trainInput.shape[3]))
    
    #lane a
    HC2MaxPool_a = keras.layers.MaxPool2D(pool_size=(3,3))(HC2min)
    HC2SepConv_a = keras.layers.SeparableConv2D(filters = 3, kernel_size = (5,5), strides=2)(HC2min)
    HC2SepConv_a = keras.layers.Conv2D(filters = int(HC2MaxPool_a.shape[3]), kernel_size=(1,1))(HC2SepConv_a)
    
    sizePad = int(HC2SepConv_a.shape[1]) - int(HC2MaxPool_a.shape[1])
    sizePad = int(sizePad/2)
    HC2MaxPool_a = keras.layers.ZeroPadding2D(padding = (sizePad,sizePad))(HC2MaxPool_a)
    outputLane_a = keras.layers.Add()([HC2SepConv_a, HC2MaxPool_a])
    
    
    #lane b
    HC2SepConv_b = keras.layers.SeparableConv2D(filters = 3, kernel_size = (3,3), strides=2)(HC2min)
    HC1Identity_b = keras.backend.eye(size = int(HC2SepConv_b.shape[1]))
    HC1Identity_b = tf.reshape(HC1Identity_b, [1, HC1Identity_b.shape[0], HC1Identity_b.shape[1], 1])
    outputLane_b = keras.layers.Add()([HC2SepConv_b, HC1Identity_b])
    
    
    finalModel = keras.models.Model(inputs=HC2min,outputs=outputLane_a)
    keras.utils.plot_model(finalModel,to_file='demo.png',show_shapes=True)
    return None
    
    
    
    
    
    