# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 23:02:05 2018

@author: Lin_Shien
"""
from selenium.webdriver.support.ui import Select
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from time import sleep
from classifier import classify_passcode
from PIL import Image
from io import BytesIO
import numpy as np
import os
import time
import requests

start_time = time.time()

mapping_list = ['2', '3', '4', '5', '7', '9', 'A', 'C', 'F', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'T', 'Y', 'Z']
la_lst = list()

try:
    browser = webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
    print(type(browser))
    browser.get('https://irs.thsrc.com.tw/IMINT?locale=tw')
except NoSuchElementException:
    browser = webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
    print(type(browser))
    browser.get('https://irs.thsrc.com.tw/IMINT?locale=tw')
    
numOfSucces = 0


agreement = browser.find_element_by_id('btn-confirm')
agreement.click()
sleep(0.01)    
    
while True:
  #session = requests.Session()                   # 利用Session能跨請求並保持參數，它也会在同一个 Session 实例发出的所有请求之间保持 cookie所以如果你向同一主机发送多个请求，底层的 TCP 连接将会被重用，从而带来显著的性能提升
  #response = session.get('https://irs.thsrc.com.tw/IMINT?locale=tw', cookies = {'from-my': 'browser'})

  
  
  origin_optionList = browser.find_element_by_xpath('.//span/select[@name="selectStartStation"]')
  destination_optionList = browser.find_element_by_xpath('.//select[@name="selectDestinationStation"]')
  origin_select = Select(origin_optionList)
  destination_select = Select(destination_optionList)
  origin_select.select_by_index(6)
  destination_select.select_by_index(7)
  sleep(0.01)
  
  

  carForm = browser.find_element_by_xpath('.//span[@id = "BookingS1Form_trainCon_trainRadioGroup"]/input')    # 選車廂
  carForm.click()
  sleep(0.01)                 # 程式太快會出錯
  
  '''
  seatPreference = browser.find_element_by_xpath('.//tr[@id = "BookingS1Form_seatCon"]/td/input[@id = "seatRadio1"]')    # 選座位喜好
  seatPreference.click()
  sleep(0.01)                                        
  '''
  
  booking_method = browser.find_element_by_xpath('.//span[@id = "BookingS1Form_bookingMethod"]/label[@for = "bookingMethod_0"]')    # 定位方式
  booking_method.click()
  sleep(0.01)   

  booking_date = browser.find_element_by_xpath('.//span[@id = "toDate"]/input[@id = "toTimeInputField"]')    # 定位日期
  booking_date.send_keys("\b" * 10)
  booking_date.send_keys("2019/7/25")
  sleep(0.01) 

  '''
  toTrain_ID = browser.find_element_by_xpath('.//span[@id = "toTrainID"]/input[@name = "toTrainIDInputField"]')    # 輸入車次
  toTrain_ID.send_keys("1000")
  sleep(0.1)
  
  returnCheckBox = browser.find_element_by_xpath('.//input[@id = "returnCheckBox"]')
  returnCheckBox.click()
  sleep(0.1)

  return_date = browser.find_element_by_xpath('.//span[@id = "backDate"]/input[@id = "backTimeInputField"]')
  return_date.send_keys("\b" * 10)
  return_date.send_keys("2018/8/30")
  sleep(0.1)

  backTrain_ID = browser.find_element_by_xpath('.//span[@id = "backTrainID"]/input[@name = "backTrainIDInputField"]')    # 輸入車次
  backTrain_ID.send_keys("1040")
  sleep(0.1)
  '''

  booking_method = browser.find_element_by_xpath('.//span[@id = "BookingS1Form_bookingMethod"]/label[@for = "bookingMethod_0"]')    # 定位方式
  booking_method.click()
  sleep(0.1)   

  bookingTime_List = browser.find_element_by_xpath('.//span[@id = "toTimeTable"]/select[@name="toTimeTable"]')    # 訂票時間
  time_select = Select(bookingTime_List)
  time_select.select_by_index(30)
  '''
  returnCheckBox = browser.find_element_by_xpath('.//input[@id = "returnCheckBox"]')        # 回程票box
  returnCheckBox.click()
  sleep(0.1)

  backTime_List = browser.find_element_by_xpath('.//span[@id = "backTimeTable"]/select[@name = "backTimeTable"]')    # 訂票時間
  backTime_select = Select(backTime_List)
  backTime_select.select_by_index(35)
  sleep(0.1)
'''
  adultTicket_NumList = browser.find_element_by_xpath('.//span[@class = "PR10"]/select[@name = "ticketPanel:rows:0:ticketAmount"]')  # 全票數
  adultTicketNum_select = Select(adultTicket_NumList)
  adultTicketNum_select.select_by_index(0)
  sleep(0.01)

  kidTicket_NumList = browser.find_element_by_xpath('.//span[@class = "PR10"]/select[@name = "ticketPanel:rows:1:ticketAmount"]')  # 兒童數
  kidTicketNum_select = Select(kidTicket_NumList)
  kidTicketNum_select.select_by_index(0)
  sleep(0.01)

  priorityTicket_NumList = browser.find_element_by_xpath('.//span[@class = "PR10"]/select[@name = "ticketPanel:rows:2:ticketAmount"]') # 愛心票
  priorityTicketNum_select = Select(priorityTicket_NumList)
  priorityTicketNum_select.select_by_index(1)
  sleep(0.01)

  elderTicket_NumList = browser.find_element_by_xpath('.//span[@class = "PR10"]/select[@name = "ticketPanel:rows:3:ticketAmount"]')
  elderTicketNum_select = Select(elderTicket_NumList)
  elderTicketNum_select.select_by_index(0)
  sleep(0.01)


  # 截圖，並儲存驗證碼圖片
  img_element = browser.find_element_by_xpath('.//span[@class = "PR10"]/img')
  location = img_element.location                  # 驗證碼座標 
  size = img_element.size                          # 驗證碼大小
  screen_shot = browser.get_screenshot_as_png()    # 回傳bytes type data
  img = Image.open(BytesIO(screen_shot))           # Image object
  # 完整座標
  left = location['x']
  top = location['y']             # 與原始碼element的座標有誤差
  right = left + size['width']
  bottom = top + size['height'] 
  # crop，截圖下來會有點模糊
  img = img.crop((left, top, right + 1, bottom + 1))
  img.save(r"C:\Users\Lin_Shien\Desktop\project\sele\ps_code.png")


  pass_code_input = browser.find_element_by_xpath('.//span[@class = "PR10"]/input[@name = "homeCaptcha:securityCode"]')  
  str_p = classify_passcode(r"C:\Users\Lin_Shien\Desktop\project\sele\ps_code.png")
  pass_code_input.send_keys("\b" * 4)
  pass_code_input.send_keys(str_p)
  sleep(0.01)

  submit_BookingReq = browser.find_element_by_xpath('.//td[@align = "right"]/input[@id = "SubmitButton"]') 
  submit_BookingReq.click()
  sleep(0.6)
    
  passed = False
  
  try:
    print("checked")
    error_div = browser.find_element_by_xpath('.//div[@id = "error"]/span/ul')
  except NoSuchElementException:
    passed = True
    
  sleep(0.01)

  if passed:
    print("-------------passed------------")
    break
      
  #os.unlink("ps_code" + str(i) + ".png")  
  browser.get('https://irs.thsrc.com.tw/IMINT?locale=tw')
  sleep(0.01)
  

"""進入訂票的第2個步驟--選車次"""  
span_tags = browser.find_elements_by_xpath('.//span')
allCarInfro = span_tags[10].text
print(allCarInfro)
#userChooseCarNum = input("請選取欲搭乘的班次(號碼依序由上往下為1開始): ")

selectCarNum = browser.find_elements_by_xpath('.//tr/td/input[@name = "TrainQueryDataViewPanel:TrainGroup"]')  

choose = selectCarNum[0]
choose.click()
sleep(0.01)

span_tags = browser.find_elements_by_xpath('.//span')
bookingInfro = span_tags[30].text
print(bookingInfro)

submit_Btn = browser.find_element_by_xpath('.//td/input[@name = "SubmitButton"]')  
submit_Btn.click()
sleep(0.01)


"""填寫訂票資訊"""
idNumInput = browser.find_element_by_xpath('.//input[@id = "idNumber"]')
idNumInput.send_keys("F111111111")
sleep(0.01)

cellphoneCheckBox = browser.find_element_by_xpath('.//input[@id = "mobileInputRadio"]')
cellphoneCheckBox.click()
sleep(0.01)

phoneNumInput = browser.find_element_by_xpath('.//input[@id = "mobilePhone"]')
phoneNumInput.send_keys("123456789")
sleep(0.01)

agreeCheckBox = browser.find_element_by_xpath('.//input[@name = "agree"]')
agreeCheckBox.click()
sleep(0.01)

end_time = time.time()

print("完成訂票")
print("共花了: " + str(end_time - start_time) + '秒')
