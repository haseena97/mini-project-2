# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:15:03 2022

@author: Acer
"""

# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import seaborn as sns
house = pd.read_csv("C:\\Users\\Acer\\Downloads\\50_Startups (2)\\50_Startups.csv",encoding='latin1')
house2 = house[['Price', 'Age_08_04','KM', 'HP','cc', 'Doors','Gears', 'Quarterly_Tax', 'Weight']]

wcat = pd.read_csv("C:\\Users\Acer\Downloads\SLR Assignment\Problem Statement (1)\delivery_time.csv")
calories = pd.read_csv("C:\\Users\Acer\Downloads\SLR Assignment\Problem Statement (1)\calories_consumed.csv")
churn = pd.read_csv("C:\\Users\Acer\Downloads\SLR Assignment\Problem Statement (1)\emp_data.csv")
hike = pd.read_csv("C:\\Users\Acer\Downloads\SLR Assignment\Problem Statement (1)\Salary_Data.csv")
house.columns = ['research', 'admin','marketing','state','profit']

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

wcat.describe().columns
wcat.columns = ['delivery', 'sorting']
calories.columns = ['gain', 'calories']
churn.columns = ['salary', 'churn']
hike.columns = ['yearexp', 'salary']
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# AT
plt.bar(height = wcat.AT, x = np.arange(1,110,1)) #sbb data ada 109 row
plt.hist(wcat.AT) #histogram
plt.boxplot(wcat.AT) #boxplot

# WC
plt.bar(height = wcat.Waist, x = np.arange(1,109+1,1))
plt.hist(wcat.Waist) #histogram
plt.boxplot(wcat.Waist) #boxplot


# Scatter plot
plt.scatter(x=wcat['sorting'], y=wcat['delivery'], color='green') 
plt.xlabel("sorting ")
plt.ylabel("delivery")
plt.show

plt.scatter(x=hike['yearexp'], y=hike['salary'], color='green') 
plt.xlabel("yearexp ")
plt.ylabel("salary")
plt.show
# correlation
np.corrcoef(wcat.Waist, wcat.AT) 
sns.regplot(x=wcat['sorting'],y=wcat['delivery'])
# Import library
import statsmodels.formula.api as smf
house['norm_research'] = np.log(house.research+1)
house['norm_marketing'] = np.log(house.marketing+1)
house['norm_research'].hist()
house['norm_marketing'].hist()
# Simple Linear Regression --line of best fit with smallest distance between points and line
model = smf.ols('profit ~ (marketing)', data = house).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(wcat['Waist']))
pred1 #dpt nilai y based in model y=mx+c
plt.scatter(x=wcat['Weight'], y=wcat['Price'], color='blue') 
x = wcat.Waist
y = pred1
plt.plot(x, y)
# Error calculation
res1 = wcat.AT - pred1 #nilai error utk y baru compare dgn y actual 
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

house2['age_poly'] = house2.Age_08_04*house2.Age_08_04
######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at -- jadikan x ada log
plt.scatter(x=(house['Marketing Spend']),y=house['Profit'],color='brown')
np.corrcoef(np.log(house2.Weight), house2.Price) #correlation
# dptkan equation model baru based on transformed log.Waist
model2 = smf.ols('salary ~ np.log(yearexp)',data = hike).fit()
model2.summary()
# prediction baru utk y-value ikut equation yg dpt dari log.Waist
pred2 = model2.predict(pd.DataFrame(wcat['Waist']))
x = wcat.Waist
y = pred2
plt.plot(x, y)
 #value dpt makin dekat dgn actual
# Error calculation
res2 = wcat.AT - pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at) --log dkat y-axis

plt.scatter(x=wcat['Waist'], y=np.log(wcat['AT']),color='orange')
np.corrcoef(wcat.Waist, np.log(wcat.AT)) #correlation

model3 = smf.ols('np.log(salary) ~ yearexp',data = hike).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at = np.exp(pred3)
pred3_at

x = wcat.Waist
y = pred3_at
plt.plot(x, y)
# value y prediction jadi overfit pulak
# Error calculation
res3 = wcat.AT - pred3_at
res_sqr3 = res3*res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('(profit) ~ (marketing)', data = house).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(wcat))
pred4_at = np.exp(pred4)
pred4_at
x = wcat.Waist
y = pred4_at
plt.plot(x, y)
# Error calculation
res4 = wcat.AT - pred4_at
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

