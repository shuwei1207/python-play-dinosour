# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:40:12 2018

@author: USER
"""
from mss import mss
import cv2 as cv
import numpy as np
from time import time
from pyautogui import position, size
import pyautogui

b=0 
dino = cv.imread('dino.png',cv.IMREAD_COLOR) 
tree = cv.imread('tree.png',cv.IMREAD_COLOR)
stree= cv.imread('stree.png',cv.IMREAD_COLOR)
bird =cv.imread('bird.png',cv.IMREAD_COLOR)
frames = 0 # record frame number
bbox={'left': 340, 'top': 114, 'width': 400, 'height': 200} # detect box
begin = time() # start time
while True:
    bbox['left'], bbox['top'] = position()  
    bbox['left'] = min(size()[0]-bbox['width']-1,bbox['left'])
    bbox['top'] = min(size()[1]-bbox['height']-1,bbox['top'])
    with mss() as sct:
        mss_im = sct.grab(bbox)
    im = np.array(mss_im.pixels,np.uint8)
    
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR) 
    result1 = cv.matchTemplate(im, dino, cv.TM_CCOEFF_NORMED) 
    _minVal, maxVal, _minLoc, maxLoc = cv.minMaxLoc(result1, None)

    threshold = 0.7
    loc1 = np.where( result1 >= threshold)
    for pt in zip(*loc1[::-1]):
        cv.rectangle(im, pt, (pt[0] + dino.shape[0], pt[1] + dino.shape[1]), (255,0,0), 2)
    # end Multiple matching
    
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR) 
    result2 = cv.matchTemplate(im, tree, cv.TM_CCOEFF_NORMED)
    _minVal, maxVal, _minLoc, maxLoc = cv.minMaxLoc(result2, None)

    threshold = 0.7
    loc2 = np.where( result2 >= threshold)
    for pt in zip(*loc2[::-1]):
        cv.rectangle(im, pt, (pt[0] + tree.shape[0], pt[1] + tree.shape[1]), (0,255,0), 2)
        b=1
    
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR) 
    result3 = cv.matchTemplate(im, stree, cv.TM_CCOEFF_NORMED)  
    _minVal, maxVal, _minLoc, maxLoc = cv.minMaxLoc(result3, None)

    threshold = 0.7
    loc3 = np.where( result3 >= threshold)
    for pt in zip(*loc3[::-1]):
        cv.rectangle(im, pt, (pt[0] + stree.shape[0], pt[1] + stree.shape[1]), (0,255,0), 2)
        b=1
        
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR) 
    result4 = cv.matchTemplate(im, bird, cv.TM_CCOEFF_NORMED) 
    _minVal, maxVal, _minLoc, maxLoc = cv.minMaxLoc(result4, None)

    threshold = 0.7
    loc4 = np.where( result4 >= threshold)
    for pt in zip(*loc4[::-1]):
        cv.rectangle(im, pt, (pt[0] + bird.shape[0], pt[1] + bird.shape[1]), (0,255,0), 2)  
        b=1
    
    cv.imshow('bbox', im)
    if 13 == cv.waitKey(1):
        break
    if b==1:
        pyautogui.press('space',1)
    b=0
end = time()
print(frames/(end-begin)) # frame per second
cv.destroyAllWindows()