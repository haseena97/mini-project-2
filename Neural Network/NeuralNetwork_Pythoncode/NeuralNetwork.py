########################## Neural Network for predicting continuous values ###############################
import numpy as np
import pandas as pd

pip install tensorflow
pip install keras

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

# Reading data 
Concrete = pd.read_csv("C:\\Users\\Acer\\Downloads\\Neural Network\\concrete\\concrete.csv")
Concrete.head()

from sklearn.model_selection import train_test_split

X = Concrete.drop(["strength"], axis=1)
Y = Concrete["strength"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
# kita nak neural network follow sequence from input to output
cont_model = Sequential()
cont_model.add(Dense(50, input_dim=8, activation="relu"))
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
