#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 08:52:23 2019

@author: keshavbachu
"""
import keras

def tranInceptionModel(trainInput, trainOutput, possibleOutcomes):
    Xvals = keras.models.Input((trainInput.shape[1], trainInput.shape[2], trainInput.shape[3]))
    
    #use 4 diff convolutions for model
    #lane 1
    lane1 = keras.Sequential()
    lane1.add(keras.layers.Conv2D(filters = 2, kernel_size = (1,1), activation='relu'))
    lane1Out = lane1(Xvals)
    lane1Flat = keras.layers.Flatten()(lane1Out)
    
    #lane2
    lane2 = keras.Sequential()
    lane2.add(keras.layers.MaxPool2D())
    lane2.add(keras.layers.Conv2D(filters = 2, kernel_size=(1,1)))
    lane2Out = lane2(Xvals)
    lane2Flat = keras.layers.Flatten()(lane2Out)
    
    #lane3
    lane3 = keras.Sequential()
    lane3.add(keras.layers.Conv2D(filters = 2, kernel_size = (1,1)))
    lane3Out = lane3(Xvals)
    
    lane3a = keras.layers.Conv2D(filters = 2, kernel_size = (1,3))
    lane3b = keras.layers.Conv2D(filters = 2, kernel_size = (3,1))
    
    lane3aOut = lane3a(lane3Out)
    lane3bOut = lane3b(lane3Out)
    
    lane3aFlat = keras.layers.Flatten()(lane3aOut)
    lane3bFlat = keras.layers.Flatten()(lane3bOut)
    
    #lane4
    lane4 = keras.Sequential()
    lane4.add(keras.layers.Conv2D(filters = 2, kernel_size=(1,1)))
    lane4.add(keras.layers.Conv2D(filters= 2, kernel_size=(3,3)))
    lane4Out = lane4(Xvals)
    
    lane4a = keras.layers.Conv2D(filters=2, kernel_size=(1,3))
    lane4b = keras.layers.Conv2D(filters=2, kernel_size=(3,1))
    
    lane4aOut = lane4a(lane4Out)
    lane4bOut = lane4b(lane4Out)
    
    lane4aFlat = keras.layers.Flatten()(lane4aOut)
    lane4bFlat = keras.layers.Flatten()(lane4bOut)
    
    #Post fuse model
    model_concat = keras.layers.concatenate([lane1Flat, lane2Flat, lane3aFlat, lane3bFlat, lane4aFlat, lane4bFlat])
    
    #FC dense layers
    dense = keras.Sequential()
    dense.add(keras.layers.Dense(2048))
    dense.add(keras.layers.Dense(256))
    dense.add(keras.layers.Dense(possibleOutcomes))
    
    final_out = dense(model_concat)
    
    #Generate model together
    #modelFuse = keras.models.Model(inputs=[],outputs=[A3,B3])
    model_fused = keras.models.Model(inputs=Xvals,outputs=final_out)
    
    #view of model
    #keras.utils.plot_model(model_fused,to_file='demo.png',show_shapes=True)
    return None