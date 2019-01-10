#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:01:26 2018

@author: keshavbachu
"""
examplesEach = 3
pictureX = 500
pictureY = 500

import numpy as np
import pandas as pd
import cv2
import os
import ImageNetwork
import InceptionModel
import PNASModel
from matplotlib import pyplot as plt


#load all file paths
file = 'Data Folder/' 
allFiles = os.listdir(file)
picturePaths = []
if '.DS_Store' in allFiles:
    allFiles.remove('.DS_Store')
for folder in allFiles:
    fileTemp = file + folder + "/Pictures/"
    print(fileTemp)
    
    #figure out a way to do this without a loop, possibly broadcast the string
    tempDir = os.listdir(fileTemp)
    
    if '.DS_Store' in tempDir:
        tempDir.remove('.DS_Store')
    for i in range(len(tempDir)):
        tempDir[i] = fileTemp + tempDir[i]
    picturePaths.append(tempDir)
#im = cv2.imread(allFiles[0], flags = cv2.IMREAD_COLOR)
    
#get all the pictures into sets
pictureHolder = []
pictureOutput = []
pictureCode = 0

for folder in picturePaths:
    counter = 0
    for path in folder:
        print("\t", path)
        
        picture = cv2.imread(path, flags = cv2.IMREAD_ANYCOLOR)
        picture = cv2.resize(picture, (pictureX, pictureY))
        picture = cv2.GaussianBlur(picture,(15,15), 0)
        pictureHolder.append(np.asanyarray(picture))
        counter += 1
        
        pictureOutput.append(pictureCode)
        
        #maintain needed size
        if(counter >= examplesEach):
            pictureCode += 1
            break

#numpy array with dimensions [# examples, pictureX, pictureY, 3]    
pictureHolder = np.asanyarray(pictureHolder)
pictureOutput = np.asanyarray(pictureOutput)
pictureOutput = pictureOutput.reshape([pictureOutput.shape[0], 1])
numClasses = pictureCode

shuffle = np.random.permutation(pictureOutput.shape[0])
pictureOutput = pictureOutput[shuffle]
pictureHolder = pictureHolder[shuffle]


#ImageNetwork.quickModel(pictureHolder, pictureOutput, numClasses)
#InceptionModel.tranInceptionModel(pictureHolder, pictureOutput, numClasses)
PNASModel.trainPNAS(pictureHolder, pictureOutput, numClasses)