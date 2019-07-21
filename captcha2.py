import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from eliminate_curve import eliminateCurve, deleCurve_and_create_pieces

_letter_cases = "qmnz" # 小写字母，去除可能干扰的i，l，o，z
_upper_cases = _letter_cases.upper() # 大写字母
_numbers = ''.join(map(str, [2])) # 数字
init_chars = ''.join(( _upper_cases, _numbers))

def create_validate_code(size=(130, 45),
                         chars=init_chars,
                         img_type="PNG",
                         mode="RGB",
                         bg_color=(0, 0, 0),
                         fg_color=(255, 255, 255),
                         font_size=30,
                         font_type=r'AdobeGothicStd-Bold.otf',
                         length=4,
                         draw_lines=False,
#                         n_line=(1, 2),
                         draw_points=True,
                         point_chance = 0):
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

    def create_lines():
        '''绘制干扰线'''
        line_num = 1 # 干扰线条数

        for i in range(line_num):
            # 起始点
            begin = (0, random.randint(10, 25))
            #结束点
            end = (130, begin[1] - 10)           
            if end[1] < 0 :
                end[1] = 0
            draw.line([begin, end], fill=(255, 255, 255), width = 5)
            return [begin, end]

    def create_points():
        '''绘制干扰点'''
        chance = min(100, max(0, int(point_chance))) # 大小限制在[0, 100]
       
        for w in range(width):
            for h in range(height):
                tmp = random.randint(0, 100)
                if tmp > 100 - chance:
                    draw.point((w, h), fill=(255, 255, 255))

    def create_strs():
        '''绘制验证码字符'''
        c_chars = get_chars()
        strs = ' %s ' % ''.join(c_chars) # 每个字符前后以空格隔开
       
        font = ImageFont.truetype(font_type, font_size)
        font_width, font_height = font.getsize(strs)

        draw.text(((width - font_width) / 3, (height - font_height) / 3),
                    strs, font=font, fill=fg_color)
       
        return ''.join(c_chars)

    if draw_lines:
        create_lines()
    if draw_points:
        create_points()
    strs = create_strs()

    # 图形扭曲参数
    params = [1 - float(random.randint(1, 2)) / 500,
              0,
              0,
              0,
              1 - float(random.randint(1, 5)) / 500,
              float(random.randint(1, 2)) / 2500,
              0.001,
              float(random.randint(1, 2)) / 2500
              ]
    img = img.transform(size, Image.PERSPECTIVE, params) # 创建扭曲


    img = np.array(img)
    #XOR

    img2 = np.zeros((45,130), np.uint8) 
    img2.fill(0) 
    
    base = random.randint(5, 15)
    # 產生干擾線
    for x in range(0,130):
        y = base + (x-130) * (x-130)*0.001
        img2[int(y)][x]=255
        img2[int(y)+1][x]=255
        img2[int(y)+2][x]=255
        #img2[int(y)+3][x]=255
        img2[int(y)-1][x]=255
        img2[int(y)-2][x]=255
        #img2[int(y)-3][x]=255
        
    for x in range(0,130):
        for y in range(0,45):
            if img[y][x][0] == img2[y][x]:
                img[y][x][0] = 0
            else:
                img[y][x][0] = 255
                
    
    return img.copy(), strs

if __name__ == "__main__":
    mapping_list = ['2', '3', '4', '5', '7', '9', 'A', 'C', 'F', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'T', 'Y', 'Z']

    lst = list()
    la = list()
    for i in range(15000):
        print(i)
        code_img, strs = create_validate_code()
        cv2.imwrite('validate2.png', code_img[:, :, 0:1])
        newimg = eliminateCurve('validate2.png')    
        p1 ,p2, p3, p4 = deleCurve_and_create_pieces(newimg)
        lst.append(cv2.resize(p1, dsize = (30, 45), interpolation=cv2.INTER_CUBIC))
        lst.append(cv2.resize(p2, dsize = (30, 45), interpolation=cv2.INTER_CUBIC))
        lst.append(cv2.resize(p3, dsize = (30, 45), interpolation=cv2.INTER_CUBIC))
        lst.append(cv2.resize(p4, dsize = (30, 45), interpolation=cv2.INTER_CUBIC))
        la1, la2, la3, la4 = strs
        la.append(mapping_list.index(la1))
        la.append(mapping_list.index(la2))
        la.append(mapping_list.index(la3))
        la.append(mapping_list.index(la4))
    
    arr = np.array(lst)
    np.save("QMN2Z_data.npy", arr)
    la = np.array(la)
    np.save("QMN2Z_label.npy", la)

    
