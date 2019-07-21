# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:53:34 2018

@author: Lin_Shien
"""

import numpy as np
import cv2
from eliminate_curve import eliminateCurve, deleCurve_and_create_pieces

l5 = np.load("HihgRail_labels.npy")

for i in range(l5.shape[0]):
  if l5[i] >= 17:
    l5[i] = l5[i] - 1
  
  