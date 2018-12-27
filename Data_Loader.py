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

for folder in picturePaths:
    counter = 0
    for path in folder:
        print("\t", path)
        
        picture = cv2.imread(path, flags = cv2.IMREAD_ANYCOLOR)
        picture = cv2.resize(picture, (pictureX, pictureY))
        pictureHolder.append(np.asanyarray(picture))
        counter += 1
        
        #maintain needed size
        if(counter > examplesEach):
            break

#numpy array with dimensions [# examples, pictureX, pictureY, 3]    
pictureHolder = np.asanyarray(pictureHolder)