#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:10:26 2018

@author: keshavbachu
"""

def sizeChecker(npList):
    size = None
    for i in npList:
        if(i.shape() != size):
            size = i.shape()
            print(size)