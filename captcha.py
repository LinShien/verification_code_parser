# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 09:08:18 2018

@author: katie
"""

##############################################################################
''' #1.產生高鐵驗證碼、處理圖片、上標籤 '''
##############################################################################
from __future__ import print_function
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
import random
from PIL import  ImageDraw, ImageFont, ImageFilter
import copy

_letter_cases = "abcdefghjkmnpqrstuvwxy" # 小写字母，去除可能干扰的i，l，o，z
_upper_cases = _letter_cases.upper() # 大写字母
_numbers = ''.join(map(str, range(3, 10))) # 数字
init_chars = ''.join(( _upper_cases, _numbers))
pic_list=[]   #RGB高鐵圖片
new_pic_list=[]   #處理完高鐵圖片
label_list=[]   #四字lebel
all_label_list=[]   #單字lebel
train_label_list=[]
test_label_list=[]

total_number = 8000
train_number = 6000
#train_number = int(total_number - (total_number / 6 ))

def create_validate_code(size=(120, 30),
                         chars=init_chars,
                         img_type="PNG",
                         mode="RGB",
                         bg_color=(192, 192, 192),
                         fg_color=(0, 0, 0),
                         font_size=24,
                         font_type=r'C:\Windows\Fonts\AdobeGothicStd-Bold.otf',
                         length=4,
                         draw_lines=True,
#                         n_line=(1, 2),
                         draw_points=True,
                         point_chance = 2):
    '''
    @todo: 生成验证码图片
    @param size: 图片的大小，格式（宽，高），默认为(120, 30)
    @param chars: 允许的字符集合，格式字符串
    @param img_type: 图片保存的格式，默认为GIF，可选的为GIF，JPEG，TIFF，PNG
    @param mode: 图片模式，默认为RGB
    @param bg_color: 背景颜色，默认为白色
    @param fg_color: 前景色，验证码字符颜色，默认为蓝色#0000FF
    @param font_size: 验证码字体大小
    @param font_type: 验证码字体，默认为 ae_AlArabiya.ttf
    @param length: 验证码字符个数
    @param draw_lines: 是否划干扰线
    @param n_lines: 干扰线的条数范围，格式元组，默认为(1, 2)，只有draw_lines为True时有效
    @param draw_points: 是否画干扰点
    @param point_chance: 干扰点出现的概率，大小范围[0, 100]
    @return: [0]: PIL Image实例
    @return: [1]: 验证码图片中的字符串
    '''

    width, height = size # 宽， 高
    img = Image.new(mode, size, bg_color) # 创建图形
    draw = ImageDraw.Draw(img) # 创建画笔

    def get_chars():
        '''生成给定长度的字符串，返回列表格式'''
        return random.sample(chars, length)

    def create_points():
        '''绘制干扰点'''
        chance = min(100, max(0, int(point_chance))) # 大小限制在[0, 100]
       
        for w in range(width):
            for h in range(height):
                tmp = random.randint(0, 100)
                if tmp > 100 - chance:
                    draw.point((w, h), fill=(0, 0, 0))

    def create_strs():
        '''绘制验证码字符'''
        c_chars = get_chars()
        strs = ' %s ' % ''.join(c_chars) # 每个字符前后以空格隔开
       
        font = ImageFont.truetype(font_type, font_size)
        font_width, font_height = font.getsize(strs)

        draw.text(((width - font_width) / 3, (height - font_height) / 3),
                    strs, font=font, fill=fg_color)
      
        return ''.join(c_chars)
    if draw_points:
        create_points()
    strs = create_strs()
#    print(strs)
    label_list.append(strs)
#    # 图形扭曲参数
#    params = [1 - float(random.randint(1, 2)) / 100,
#              0,
#              0,
#              0,
#              1 - float(random.randint(1, 10)) / 100,
#              float(random.randint(1, 2)) / 500,
#              0.001,
#              float(random.randint(1, 2)) / 500
#              ]
#    img = img.transform(size, Image.PERSPECTIVE, params) # 创建扭曲

#    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE) # 滤镜，边界加强（阈值更大）

    return img, strs
    
 
for i in range(0,total_number) :   
    if __name__ == "__main__":
        code_img = create_validate_code()[0]
        code_img.save("validate.png", "PNG")
    img1 = cv2.imread('validate.png')
    #灰階&二進位原圖
    img1_gray= cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img1_gray, 127, 255, cv2.THRESH_BINARY_INV)
#    plt.imshow(thresh)
#    plt.show()   
       
    ##########################################################################
    #二進位弧線
    img2 = np.zeros((30,120), np.uint8)     
    #fill the image with black  
    img2.fill(0) 
#    img2[1][2]=0
    for x in range(0,120):
        y=10+(x-120)*(x-120)*0.0007
        img2[int(y)][x]=255
        img2[int(y)+1][x]=255
        img2[int(y)+2][x]=255
#        img2[int(y)+3][x]=255
        img2[int(y)-1][x]=255
        img2[int(y)-2][x]=255
#        img2[int(y)-3][x]=255
             
    #XOR
    for x in range(0,120):
        for y in range(0,30):
            if thresh[y][x]==img2[y][x]:
                thresh[y][x]=0
            else:
                thresh[y][x]=255
    plt.imshow(thresh,cmap=plt.cm.gray)
    plt.axis('off')
#    plt.show()
    
    plt.savefig("pic.png",bbox_inches = 'tight')
    plt.close()
    
    img3 = Image.open("pic.png")
    #剪裁
#    plt.imshow(im)
    region = img3.crop((24,10,359,93))
    img3.close()
#    print(type(region))
#    region.save("test1.jpeg")
    
    #改變大小
#    img = Image.open("test1.jpeg")
    img=region.resize((120,30),Image.ANTIALIAS)
    img.save("pict.png")
    im = cv2.imread("pict.png")
    pic_list.append(im)  #儲存pic_list為RGB
#    plt.imshow(im)
    
