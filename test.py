# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 23:31:57 2018

@author: Lin_Shien
"""

import numpy as np
import keras
import cv2
from eliminate_curve import eliminateCurve
from classifier import sort, readImageAndCreateRect, sortAllRect, classify_passcode


img_array = cv2.imread('testingImg.png')
#img_array = np.stack(())
clone_unchanged = img_array.copy()                                         # 用來locate的，不能修改或在上面標輪廓和長方形

gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)                                            # 轉成灰階圖
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)                           # 轉成二值圖
im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # 畫出輪廓
    
clone = img_array.copy()                                                   # mutable
cv2.drawContours(clone, contours[: len(contours)], -1, (0, 255, 0), 0)     # clone會被原地修改
#cv2.imwrite("4con2.jpg", clone)
    
coord_list = list()
coord_x_list = list()                              # 用 x座標來記住字母順序

for cnt in contours[: len(contours)] :
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 255, 0), 2)
    coord_list.append((x, y, w, h))
    coord_x_list.append(x)