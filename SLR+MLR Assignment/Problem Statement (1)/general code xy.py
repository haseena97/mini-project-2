# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:34:08 2022

@author: Acer
"""

import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt
wcat = pd.read_csv("C:\\Users\Acer\Downloads\Simple Linear Regression\wcat\wcat.csv")
x = wcat.Waist
y = wcat.AT
wcat.describe()
plt.scatter(x,y, color='green') 
np.corrcoef(x,y)
