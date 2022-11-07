# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:50:17 2022

@author: Acer
"""

# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

wcat = pd.read_csv("C:\\Users\Acer\Downloads\Problem Statement (1)\calories_consumed.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

wcat.describe()
wcat.columns
# kalau nak tukar nama header column
#wcat.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
wcat.rename(columns={'Calories ': 'calories'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import seaborn as sns
# gain
plt.bar(height = wcat.gain, x = np.arange(1,15,1)) #sbb data ada 109 row
plt.hist(wcat.gain) #histogram
plt.boxplot(wcat.gain) #boxplot
sns.distplot(wcat.gain)
# calories
plt.bar(height = wcat.calories, x = np.arange(1,15,1))
plt.hist(wcat.calories) #histogram
plt.boxplot(wcat.calories) #boxplot
sns.distplot(wcat.calories)

# Scatter plot
plt.scatter(x=wcat['calories'], y=wcat['gain'], color='green') 
plt.xlabel("Waist")
plt.ylabel("AT")
plt.show
# correlation
np.corrcoef(wcat.calories, wcat.gain) 

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression --line of best fit with smallest distance between points and line
# model = smf.ols(Y~X)
model = smf.ols('gain ~ calories', data = wcat).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(wcat['calories']))
pred1 #dpt nilai y based in model y=mx+c
plt.scatter(x=wcat['Waist'], y=wcat['AT'], color='blue') 
x = wcat.Waist
y = pred1
plt.plot(x, y)
# Error calculation
res1 = wcat.gain - pred1 #nilai error utk y baru compare dgn y actual 
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at -- jadikan x ada log
plt.scatter(x=np.log(wcat['calories']),y=wcat['gain'],color='brown')
np.corrcoef(np.log(wcat.calories), wcat.gain) #correlation
# dptkan equation model baru based on transformed log.Waist
model2 = smf.ols('gain ~ np.log(calories)',data = wcat).fit()
model2.summary()
# prediction baru utk y-value ikut equation yg dpt dari log.Waist
pred2 = model2.predict(pd.DataFrame(wcat['calories']))
x = wcat.calories
y = pred2
plt.plot(x, y)
 #value dpt makin dekat dgn actual
# Error calculation
res2 = wcat.gain - pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at) --log dkat y-axis

plt.scatter(x=wcat['calories'], y=np.log(wcat['gain']),color='orange')
np.corrcoef(wcat.calories, np.log(wcat.gain)) #correlation

model3 = smf.ols('np.log(gain) ~ calories',data = wcat).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(wcat['calories']))
pred3_at = np.exp(pred3)
pred3_at

x = wcat.Waist
y = pred3_at
plt.plot(x, y)
# value y prediction jadi overfit pulak
# Error calculation
res3 = wcat.gain - pred3_at
res_sqr3 = res3*res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(gain) ~ calories + I(calories*calories)', data = wcat).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(wcat))
pred4_at = np.exp(pred4)
pred4_at
x = wcat.Waist
y = pred4_at
plt.plot(x, y)
# Error calculation
res4 = wcat.gain - pred4_at
res_sqr4 = res4*res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse=pd.DataFrame(data)
table_rmse

###################
# The best model
from sklearn.model_selection import train_test_split

train, test = train_test_split(wcat, test_size = 0.2)

# fit model dekat training data 
finalmodel = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.AT - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.AT - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

