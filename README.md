# Verification Code Parser with Simple CNN
***
此專案是針對某些網站的驗證碼去做判斷，能使爬蟲程式深入某些需要驗證碼的網站，意可延伸到自動登入程式或是電商自動訂購系統，此 CNN 使用的訓練資料利用往也爬蟲得到，資且且經過某些特殊處理。

![img1](https://github.com/LinShien/verification_code_parser/blob/master/fig/test_img.png)
***
## Detail and Steps
---
### Step 1 to Step 4 are data pre-processing steps(資料預處理)  
 1. First, use functions provided by OpenCV lib to denoise the verification code image to avoid the noise to affect our classifying  correctness.  The result would be like this:  
  ![img2](https://github.com/LinShien/verification_code_parser/blob/master/fig/denoised_img.png)    
    
    For more details about denosing technique: [non-local denosing](https://zh.wikipedia.org/wiki/%E9%9D%9E%E5%B1%80%E9%83%A8%E5%B9%B3%E5%9D%87)

 2. Second, use threshold method(also provided by OpenCV lib) to turn the intensity of image into only binary.
   The result would be like this:  
   ![img3](https://github.com/LinShien/verification_code_parser/blob/master/fig/threshold_img.png)  
     
     For more details about intensity thresholding, you can check: [圖片二值化
   ](https://zh.wikipedia.org/wiki/%E4%BA%8C%E5%80%BC%E5%8C%96)
   
 3. Third, use [eliminate_curve.py](https://github.com/LinShien/verification_code_parser/blob/master/model/eliminate_curve.py) to eliminate the curve contained in a verification code, result is below like this:  
 ![img4](https://github.com/LinShien/verification_code_parser/blob/master/fig/result.png)  
   
     The details about how we eliminate the curve is that we use [nonlinear regression(非線性回歸)](https://en.wikipedia.org/wiki/Nonlinear_regression) , and the actual implementation is function LinearRegression() and PolynomialFeatures() with degree = 2 provided by sklearn lib
   
 4. Then, you can use [eliminate_curve.py](https://github.com/LinShien/verification_code_parser/blob/master/model/eliminate_curve.py) to cut the verification code image into 4 pieces like this:  
 ![img3](https://github.com/LinShien/verification_code_parser/blob/master/fig/p1.png)    ![img5](https://github.com/LinShien/verification_code_parser/blob/master/fig/p2.png)    ![img6](https://github.com/LinShien/verification_code_parser/blob/master/fig/p3.png)    ![img7](https://github.com/LinShien/verification_code_parser/blob/master/fig/p4.png)

### Step 5 is model trainning(模型訓練)  
  5. You can use the pieces to train your own model under the [model dirctory](https://github.com/LinShien/verification_code_parser/tree/master/model) or just test the pre-trained mode like this:  
  ![img8](https://github.com/LinShien/verification_code_parser/blob/master/fig/classify_demo.png)
---  

## Implementation of trained model(Deprecated)  
  We build a `THSR automatic booking app` based on the trained model to crack the verification code on the [THSR booking website](https://irs.thsrc.com.tw/IMINT/?locale=tw). But due to some legal issues and the website is enhanced with anti-scraping feature, we don't provide the source code of this app.  
  
  ![gif1](https://github.com/LinShien/verification_code_parser/blob/master/fig/final_demo.gif)
