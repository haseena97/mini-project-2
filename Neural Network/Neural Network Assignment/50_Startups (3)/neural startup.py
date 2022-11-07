########################## Neural Network for predicting continuous values ###############################
import numpy as np
import pandas as pd

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

# Reading data 
Concrete = pd.read_csv("C:/Users/Acer/Desktop/Data Science/Neural Network/Neural Network Assignment/50_Startups (3)/50_Startups.csv")
Concrete.head()
Concrete.columns = ['research', 'admin','marketing','state','profit']

Concrete['norm_research'] = np.log(Concrete.research+1)
Concrete['norm_marketing'] = np.log(Concrete.marketing+1)
# encode state
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
label_encoders['state'] = LabelEncoder()
Concrete['state'] = label_encoders['state'].fit_transform(Concrete['state']) 

from sklearn.model_selection import train_test_split

X = Concrete.drop(['profit'], axis=1)
Y = Concrete["profit"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
# kita nak neural network follow sequence from input to output
cont_model = Sequential()
cont_model.add(Dense(7, input_dim=4, activation="relu")) # input
cont_model.add(Dense(10, activation="relu")) # hidden layer
cont_model.add(Dense(1, kernel_initializer="normal")) # output
cont_model.compile(loss="mean_squared_error", optimizer = "adam", metrics = ["mse"])

model = cont_model
model.fit(np.array(X_train), np.array(y_train), epochs=6)

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


