# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:21:13 2018

@author: Lin_Shien
"""
import numpy as np
import cv2

def sort(coordinate_list, num_list) :                 # sort the all coordinate based on their area
        for i in range(0, len(coordinate_list) - 1) :
            for j in range(1, len(coordinate_list)) :
                if (Cal_area(coordinate_list[i]) <= Cal_area(coordinate_list[j])) :
                    temp1 = coordinate_list[i]
                    coordinate_list[i] = coordinate_list[j]
                    coordinate_list[j] = temp1
                    temp2 = num_list[i]
                    num_list[i] = num_list[j]
                    num_list[j] = temp2
               
def Cal_area(coordinate) :   # calculate the area of the coordinate
        return w * h

B = cv2.imread("co.png")
gray = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)                      # 轉成灰階圖
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 轉成二值圖
im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 畫出輪廓

clone = B.copy()
cv2.drawContours(clone, contours[: len(contours) - 1], -1, (0, 255, 0), 0)
cv2.imwrite("contours.jpg", clone)

coord_list = list()
num_list = list()
i = 1

for cnt in contours[: len(contours) - 1] :
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 0)
    coord_list.append((x, y, w, h))
    num_list.append(i)
    i += 1
    
sort(coord_list, num_list)

(x, y, w, h) = coord_list[0]
cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 0)
rect = clone[y : y + h + 1, x : x + w,]
cv2.imwrite("done.jpg", clone)





