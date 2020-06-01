# -*- coding: utf-8 -*-
"""
Created on Tue Feb 36 22:24:14 2019

@author: xxx
"""

import urllib.request
from urllib.request import urlretrieve
import urllib.parse
import os

from bs4 import BeautifulSoup
os.chdir('./RCar')
try:
    
    for k in range(13,24):
        
        url='https://www.shutterstock.com/search/cars+street?image_type=photo&mreleased=false&page='+str(k)#+str(j)
        #url='https://www.shutterstock.com/search/similar/600253013'
        headers = {}
        headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
        req = urllib.request.Request(url, headers = headers)
        resp = urllib.request.urlopen(req)
        print("connection ok")
        respData = resp.read()
        resp.close()
        print('Done')
        soup = BeautifulSoup(respData, "html.parser")
    
        div = soup.find_all('div',{'class':'z_h_f'})
        for i in range(0, len(div)-1):
            img = div[i].find_all('img')
            
            image_src = img[0].get('src')
            name = str(k) + str(i) + '.jpg'
            try:
                
                urlretrieve(image_src, name)
                print('Downloaded image of ' + name)
            except Exception as ee:
                print(str(ee))
                pass
        
except Exception as e:
    print('0000')
    print(str(e))
    