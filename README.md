# Verification Code Parser with Simple CNN
***
此專案是針對某些網站的驗證碼去做判斷，能使爬蟲程式深入某些需要驗證碼的網站，意可延伸到自動登入程式或是電商自動訂購系統，此 CNN 使用的訓練資料利用往也爬蟲得到，資且且經過某些特殊處理。

![img1](https://github.com/LinShien/verification_code_parser/blob/master/fig/pass_code4.png)
***
## Detail and Steps
1. First, use [eliminate_curve.py](https://github.com/LinShien/verification_code_parser/blob/master/model/eliminate_curve.py) to eliminate the curve contained in a verification code, result is below like this:

![img2](https://github.com/LinShien/verification_code_parser/blob/master/fig/result.png)

2. Second, you can use [eliminate_curve.py](https://github.com/LinShien/verification_code_parser/blob/master/model/eliminate_curve.py) to cut the verification code image into 4 pieces like this:

![img3](https://github.com/LinShien/verification_code_parser/blob/master/fig/p1.png)  ![img4](https://github.com/LinShien/verification_code_parser/blob/master/fig/p2.png)  ![img5](https://github.com/LinShien/verification_code_parser/blob/master/fig/p3.png)  ![img6](https://github.com/LinShien/verification_code_parser/blob/master/fig/p4.png)

3. You can use the pieces to train your own model under the [model dirctory](https://github.com/LinShien/verification_code_parser/tree/master/model) or just test the pre-trained mode like this:

![img7]
