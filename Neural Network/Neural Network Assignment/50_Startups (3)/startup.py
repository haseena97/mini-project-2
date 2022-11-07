# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:43:29 2022

@author: Acer
"""

########################## Neural Network for predicting continuous values ###############################
import numpy as np
import pandas as pd

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

# Reading data 
Concrete = pd.read_csv("C:\\Users\\Acer\\Downloads\\Neural Network Assignment\\50_Startups (3)\\forestfires.csv")
Concrete.head()
Concrete.columns = ['research', 'admin','marketing','state','profit']

Concrete['norm_research'] = np.log(Concrete.research+1)
Concrete['norm_marketing'] = np.log(Concrete.marketing+1)
# encode state
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
label_encoders['size_category'] = LabelEncoder()
Concrete['size_category'] = label_encoders['size_category'].fit_transform(Concrete['size_category']) 
Concrete.columns
from sklearn.model_selection import train_test_split
train_X = train.iloc[:,2:30]
X = Concrete.iloc[:,2:30]
Y = Concrete["size_category"] # output mesti encode
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
# kita nak neural network follow sequence from input to output
cont_model = Sequential()
cont_model.add(Dense(50, input_dim=28, activation="relu"))
cont_model.add(Dense(250, activation="relu"))
cont_model.add(Dense(1, kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error", optimizer = "adam", metrics = ["mse"])

model = cont_model
model.fit(np.array(X_train), np.array(y_train), epochs=20)

# On Test dataset
pred = model.predict(np.array(X_test))
pred = pd.Series([i[0] for i in pred])

# Accuracy
np.corrcoef(pred, y_test)

layerCount = len(model.layers)
layerCount

# On Train dataset
pred_train = model.predict(np.array(X_train))
pred_train = pd.Series([i[0] for i in pred_train])

np.corrcoef(pred_train, y_train) #this is just because some model's count the input layer and others don't


