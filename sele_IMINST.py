# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 23:02:05 2018

@author: Lin_Shien
"""
from selenium.webdriver.support.ui import Select
from selenium import webdriver
from time import sleep
from classifier import classify_passcode
import requests
from selenium.webdriver.common.keys import Keys 
from PIL import Image
from io import BytesIO
import numpy as np


mapping_list = ['2', '3', '4', '5', '7', '9', 'A', 'C', 'F', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'T', 'V', 'Y', 'Z']
la_lst = list()

for i in range(1000):
  #session = requests.Session()                   # 利用Session能跨請求並保持參數，它也会在同一个 Session 实例发出的所有请求之间保持 cookie所以如果你向同一主机发送多个请求，底层的 TCP 连接将会被重用，从而带来显著的性能提升
  #response = session.get('https://irs.thsrc.com.tw/IMINT/', cookies = {'from-my': 'browser'})


  browser = webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
  print(type(browser))
  browser.get('https://irs.thsrc.com.tw/IMINT?locale=tw')


  origin_optionList = browser.find_element_by_xpath('.//span/select[@name="selectStartStation"]')
  destination_optionList = browser.find_element_by_xpath('.//select[@name="selectDestinationStation"]')
  origin_select = Select(origin_optionList)
  destination_select = Select(destination_optionList)
  origin_select.select_by_index(1)
  destination_select.select_by_index(2)

  carForm = browser.find_element_by_xpath('.//span[@id = "BookingS1Form_trainCon_trainRadioGroup"]/input')    # 選車廂
  carForm.click()
  sleep(0.1)                 # 程式太快會出錯

  seatPreference = browser.find_element_by_xpath('.//tr[@id = "BookingS1Form_seatCon"]/td/input[@id = "seatRadio1"]')    # 選座位喜好
  b = seatPreference.click()
  sleep(0.1)                                        

  booking_method = browser.find_element_by_xpath('.//span[@id = "BookingS1Form_bookingMethod"]/label[@for = "bookingMethod_0"]')    # 定位方式
  booking_method.click()
  sleep(0.1)   

  booking_date = browser.find_element_by_xpath('.//span[@id = "toDate"]/input[@id = "toTimeInputField"]')    # 定位日期
  booking_date.send_keys("\b" * 10)
  booking_date.send_keys("2018/8/28")
  sleep(0.1) 

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
  time_select.select_by_index(25)
  
  returnCheckBox = browser.find_element_by_xpath('.//input[@id = "returnCheckBox"]')        # 回程票box
  returnCheckBox.click()
  sleep(0.1)

  backTime_List = browser.find_element_by_xpath('.//span[@id = "backTimeTable"]/select[@name = "backTimeTable"]')    # 訂票時間
  backTime_select = Select(backTime_List)
  backTime_select.select_by_index(29)
  sleep(0.1)

  adultTicket_NumList = browser.find_element_by_xpath('.//span[@class = "PR10"]/select[@name = "ticketPanel:rows:0:ticketAmount"]')  # 全票數
  adultTicketNum_select = Select(adultTicket_NumList)
  adultTicketNum_select.select_by_index(1)
  sleep(0.1)

  kidTicket_NumList = browser.find_element_by_xpath('.//span[@class = "PR10"]/select[@name = "ticketPanel:rows:1:ticketAmount"]')  # 兒童數
  kidTicketNum_select = Select(kidTicket_NumList)
  kidTicketNum_select.select_by_index(1)
  sleep(0.1)

  priorityTicket_NumList = browser.find_element_by_xpath('.//span[@class = "PR10"]/select[@name = "ticketPanel:rows:2:ticketAmount"]') # 愛心票
  priorityTicketNum_select = Select(priorityTicket_NumList)
  priorityTicketNum_select.select_by_index(1)
  sleep(0.1)

  elderTicket_NumList = browser.find_element_by_xpath('.//span[@class = "PR10"]/select[@name = "ticketPanel:rows:3:ticketAmount"]')
  elderTicketNum_select = Select(elderTicket_NumList)
  elderTicketNum_select.select_by_index(1)
  sleep(0.1)


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
  img.save("ps_code" + str(i) + ".png")
  ans = classify_passcode()


  pass_code_input = browser.find_element_by_xpath('.//span[@class = "PR10"]/input[@name = "homeCaptcha:securityCode"]')  
  str_p = classify_passcode()
  pass_code_input.send_keys(str_p)
  sleep(0.1)

  submit_BookingReq = browser.find_element_by_xpath('.//td[@align = "right"]/input[@id = "SubmitButton"]') 
  submit_BookingReq.click()
  
  if browser.current_url != "https://irs.thsrc.com.tw/IMINT?locale=tw":
    la1 = mapping_list.index(str_p[0])
    la2 = mapping_list.index(str_p[1])
    la3 = mapping_list.index(str_p[2])
    la4 = mapping_list.index(str_p[3])
    la_lst.append(la1)
    la_lst.append(la2)
    la_lst.append(la3)
    la_lst.append(la4)
    
  sleep(2)
  browser.close()

la = np.array(la_lst)
np.save("HighRailData_label.npy", la)