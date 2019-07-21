# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:01:19 2018

@author: Lin_Shien
"""
"""
fastNlMeansDenoising(src, h, templateWindowSize, searchWindowSize) 
h : Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes image details, 
    smaller h value preserves details but also preserves some noise.

templateWindowSize : Size in pixels of the template patch that is used to compute weights. 
    Should be odd. Recommended value 7 pixels

searchWindowSize : Size in pixels of the window that is used to compute weighted average for given pixel. 
    Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def find_bound(array, center, coord_y):     # 用來找center的上下界
    up = 0
    down = 0
    
    for i in range(1, center):            # 往上找
        if array[center - i, coord_y] != array[center, coord_y]:
            break
        up+=1
        
    for i in range(1, array.shape[0] - center):
        if array[center + i, coord_y] != array[center, coord_y]:
            break
        down+=1
    return up , down


def find_lineSeg(array, direction, length):              # brute force 要套用在2極化後的矩陣，
    if direction == 1:
        pre_coords =  np.where(array[:, 0] == 255)[0]                    # from leftmost
        if (pre_coords.shape[0] - 1) < 0 :                               # 防止抓到空座標
            pre_coord_downmost = 0
            pre_coord_upmost = 0
        else:
            pre_coord_downmost = pre_coords[pre_coords.shape[0] - 1]
            pre_coord_upmost = pre_coords[0]
            
        for i in range(1, array.shape[1]):
            downmost_coords = np.where(array[:, i] == 255)[0]             # 觀察column上最下方的255出現在哪個座標，
            if downmost_coords.shape[0] - 1 < 0:                          # 排除空的column
                downmost_coord = 0
                upmost_coord = 0
            else:
                downmost_coord = downmost_coords[downmost_coords.shape[0] - 1]
                upmost_coord = downmost_coords[0]
                
            if pre_coord_downmost != 0 and pre_coord_upmost != 0:
                if abs(pre_coord_downmost - downmost_coord) >= length or abs(pre_coord_upmost - upmost_coord) >= length:       # 如果跟上一次的座標差很多，代表交界在上一座標
                    return i - 1                                                     # return previous index
            pre_coord_downmost = downmost_coord
            pre_coord_upmost = upmost_coord
 
           
    if direction == -1:
        pre_coords =  np.where(array[:, array.shape[1] - 1] == 255)[0]            # from rightmost        
        if (pre_coords.shape[0] - 1) < 0 :
            pre_coord_downmost = 0
            pre_coord_upmost = 0
        else:
            pre_coord_downmost = pre_coords[pre_coords.shape[0] - 1]
            pre_coord_upmost = pre_coords[0]
        for i in range(1, array.shape[1]):
            downmost_coords = np.where(array[:, array.shape[1] - i] == 255)[0]
            if downmost_coords.shape[0] - 1 < 0:
                downmost_coord = 0
                upmost_coord = 0
            else:
                downmost_coord = downmost_coords[downmost_coords.shape[0] - 1]
                upmost_coord = downmost_coords[0]
                
            if pre_coord_downmost != 0 and pre_coord_upmost != 0:    
                if abs(pre_coord_downmost - downmost_coord) >= length or abs(pre_coord_upmost - upmost_coord) >= length and not (pre_coord_downmost == pre_coord_upmost == 0):
                    return array.shape[1] - i + 1                                           
            pre_coord_downmost = downmost_coord
            pre_coord_upmost = upmost_coord
    
    
def find_bound_of_letter(array, coordOfMeans):
    k = len(coordOfMeans)
    
    center_list = list()
    for i in range(k):
        y, x = coordOfMeans[i]
        x = int(round(x))
        center_list.append((x, y))
    
    letter_bounds = list()
    for i in range(k):
        if i != 0:
            left_bound = center_list[i][0] - find_lineSeg(array[:, 0 : center_list[i][0] + 1], -1, 5)
            right_bound = center_list[i][0] + find_lineSeg(array[:, center_list[i][0] : center_list[i + 1][0]], 1, 5)
            letter_bounds.append((left_bound, right_bound))
            
        if i == 0:
            left_bound = find_lineSeg(array[:, 0 : center_list[i][0] + 1], -1, 5)
            right_bound = center_list[i][0] + find_lineSeg(array[:, center_list[i][0] : center_list[i + 1][0]], 1, 5)
            letter_bounds.append((left_bound, right_bound))
    return letter_bounds
    

"""
eliminateCurve:
    read a page, and eliminate the curve or line segment on the captcha image,
    return type is a denoised image with one channel
"""
def eliminateCurve(img_name, brightness):            
    images_detect = cv2.imread(img_name)
    lenOfYcoord = images_detect.shape[1]
    #clone_unchanged = images_detect.copy()   # 用來locate的，不能修改或在上面標輪廓和長方形

    gray = cv2.cvtColor(images_detect, cv2.COLOR_BGR2GRAY)          # 先轉灰階
    gray = cv2.fastNlMeansDenoising(gray, None, brightness, 7, 21)          # 降躁 
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)  # 轉成二值圖，無原地修改
    clone = binary.copy()
    clone_unchanged = binary.copy()

    #return clone_unchanged
    binary[:, find_lineSeg(clone_unchanged, 1, 2) : find_lineSeg(clone_unchanged, -1, 2)] = 0                        # 把字母部分挖空
    line_segment = np.where(binary == 255)         # 找出線段的位置，一個tuple代表x軸，另一個代表y軸座標
    '''
    plt.scatter(line_segment[1], lenOfYcoord - line_segment[0], s = 100, c = 'red', label = 'Cluster 1')
    plt.ylim(ymin=0)  
    plt.ylim(ymax=lenOfYcoord) 
    plt.show()
    '''
    X = np.array([line_segment[1]])
    Y = lenOfYcoord - np.array(line_segment[0])                 # 換成左下方為起點的座標

    poly_reg= PolynomialFeatures(degree = 2)           # degree 2 hypothesis, 其實就是 ... + ax + b = y....
    X_ = poly_reg.fit_transform(X.T)                   # 跟data X fit => compute 就是把 x 值帶入hypothesis中
    regr = LinearRegression()                          # regreesion
    regr.fit(X_, Y) 

    X2 = np.array([[i for i in range(0, images_detect.shape[1])]])      # 預測座標
    X2_ = poly_reg.fit_transform(X2.T)                                  # 算出函數值

    newimg = clone

    for ele in np.column_stack([regr.predict(X2_).round(0), X2[0] ,]):
        pos = lenOfYcoord - int(ele[0])                                         # coordinate y (array x) 
        up, down = find_bound(newimg, pos, int(ele[1]))
        
        if (up + down + 1) < 7:                                       # 太細代表這column只有線段而已
            newimg[pos - up : pos + down + 1, int(ele[1])] = 255 - newimg[pos - up : pos + down + 1, int(ele[1])]   # slice要多+1
        else:
            newimg[pos - 2 : pos + 2, int(ele[1])] = 255 - newimg[pos - 2 : pos + 2, int(ele[1])]
    

    #newimg = cv2.fastNlMeansDenoising(newimg, None, 40, 7, 21)          # 降躁會使字母擴大 => distortion
    '''
    plt.subplot(121)
    plt.imshow(gray)
    plt.subplot(122)
    plt.imshow(newimg)
    plt.show()
    '''
    return newimg.copy()

def takeOne(elem):
    return elem[0]


def deleCurve_and_create_pieces(img):   
    h, w = img.shape          # x, y軸的大小

    Data_list = [(h - x, y) for x in range(h) for y in range(w) if img[x][y]]       # 紀錄 pixel = 255 的點座標

    Data = np.array(Data_list)   # 換成 len(Data_list) x 2 的矩陣
    n_clusters = 4               # k

    k_means = KMeans(init='k-means++', n_clusters = n_clusters)
    k_means.fit(Data)                                       # compute k means
    k_means_labels = k_means.labels_                        # 回傳每個座標屬於k means的哪一類
    k_means_cluster_centers = k_means.cluster_centers_      # 回傳 k means 的群中心 

    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF3300']
             
    #plt.figure()
    #plt.hold(True)
    
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k                          # 找除屬於k類的座標，存成bool矩陣
        cluster_center = k_means_cluster_centers[k]
        '''
        plt.plot(Data[my_members, 1], Data[my_members, 0], 'w',   # 畫出資料
             markerfacecolor = col, marker = '.')
        
        plt.plot(cluster_center[1], cluster_center[0], 'o', markerfacecolor = col,   # 畫出群中心
             markeredgecolor = 'k', markersize = 6)
        '''
    '''
    plt.title('KMeans')    
    plt.grid(True)
    plt.show()
    '''
    center_list = list()
    for i in range(k + 1):
        y, x = k_means_cluster_centers[i]
        x = int(round(x))
        center_list.append((x, y))

    center_list.sort(key = takeOne) 

    bound_r1 = int(round(abs(center_list[1][0] + center_list[0][0])) / 2) 
    bound_l1 = int(round(abs(center_list[0][0] * 2 - bound_r1)))


    bound_r2 = int(round(abs(center_list[1][0] + center_list[2][0])) / 2) 
    bound_l2 = bound_r1


    bound_r3 = int(round(abs(center_list[2][0] + center_list[3][0])) / 2)  
    bound_l3 = bound_r2


    bound_r4 = int(round(abs(center_list[3][0] * 2 - bound_r3))) 
    bound_l4 = bound_r3

    
    #cv2.imwrite('p1.png', img[:, bound_l1 : bound_r1 + 1])
    #cv2.imwrite('p2.png', img[:, bound_l2 : bound_r2 + 1])
    #cv2.imwrite('p3.png', img[:, bound_l3 : bound_r3 + 1])
    #cv2.imwrite('p4.png', img[:, bound_l4 : bound_r4 + 1])
    
    return img[:, bound_l1 : bound_r1 + 1], img[:, bound_l2 : bound_r2 + 1], img[:, bound_l3 : bound_r3 + 1], img[:, bound_l4 : bound_r4 + 1]


'''
center_list = list()
for i in range(k + 1):
    y, x = k_means_cluster_centers[i]
    x = int(round(x))
    center_list.append((x, y))

center_list.sort(key = takeOne) 
   
letter_bounds = list()
for i in range(k + 1):
       if i != 0 and i != k:
           left_bound = center_list[i][0] - find_lineSeg(newimg[:, 0 : center_list[i][0] + 1], -1, 8)
           right_bound = center_list[i][0] + find_lineSeg(newimg[:, center_list[i][0] : center_list[i + 1][0]], 1, 8)
           letter_bounds.append((left_bound, right_bound))
            
       if i == 0:
           left_bound = find_lineSeg(newimg[:, 0 : center_list[i][0] + 1], -1, 5)
           right_bound = center_list[i][0] + find_lineSeg(newimg[:, center_list[i][0] : center_list[i + 1][0]], 1, 8)
           letter_bounds.append((left_bound, right_bound))
       
       if i == k:
           left_bound = find_lineSeg(newimg[:, 0 : center_list[i][0] + 1], -1, 5)
           right_bound = center_list[i][0] + find_lineSeg(newimg[:, center_list[i][0] : newimg.shape[1]], 1, 8)
           letter_bounds.append((left_bound, right_bound))
'''
