#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:01:26 2018

@author: keshavbachu
"""
import numpy as np
import pandas as pd
import cv2
import os

#load all the files from certian diseases and import as numpy arrays with RGB channels
file = 'Data Folder/Actinic Keratosis/Pictures/' 
allFiles = os.listdir(file)
im = cv2.imread(allFiles[0], flags = cv2.IMREAD_COLOR)