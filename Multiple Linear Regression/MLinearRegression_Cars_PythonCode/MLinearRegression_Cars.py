# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
cars = pd.read_csv("C:\\Users\Acer\Downloads\Multiple Linear Regression\Cars\Cars.csv")

cars.describe()
hp_mean = cars.HP.mean
hp_median = cars.HP.median


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:,:])
                             
# Correlation matrix 
cars.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('MPG ~ WT+VOL+SP+HP', data=cars).fit() # simple regression model

# Summary
ml1.summary()

# calculating VIF's values of independent variables
rsq_hp = smf.ols('HP ~ WT+VOL+SP', data=cars).fit().rsquared  
vif_hp = 1/(1-rsq_hp) 

rsq_wt = smf.ols('WT ~ HP+VOL+SP', data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt)

rsq_vol = smf.ols('VOL ~ WT+SP+HP', data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP ~ WT+VOL+HP', data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
#kalau nak write dalam excel file (nak header=true,kalau x nak header=None,kalau x nak index=False)sep=separator
Vif_frame.to_csv('C:\\Users\\Acer\\Downloads\\Cars\\test2.csv',header=True,index=False,sep=',',mode='w')
#kalau nak write dalam txt file (nak header=true,kalau x nak header=None,kalau x nak index=False),\t bagi sengkang
Vif_frame.to_csv('C:\\Users\\Acer\\Downloads\\Cars\\test2.txt',header=True,index=False,sep='\t',mode='w')

# As weight is having higher VIF value, we are not going to include this prediction model
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(x=(cars['HP']),y=cars['MPG'],color='blue')
plt.scatter(x=(cars['WT']),y=cars['MPG'],color='red')
plt.scatter(x=(cars['VOL']),y=cars['MPG'],color='yellow')
plt.scatter(x=(cars['SP']),y=cars['MPG'],color='black')
plt.legend(loc='upper left')
plt.show()
# final model (buang WT)
#final_ml= smf.ols('MPG ~ VOL+SP+HP+I(HP*HP)', data = cars).fit()
final_ml= smf.ols('MPG ~ VOL+SP+HP', data = cars).fit()
final_ml.summary()


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train, cars_test  = train_test_split(cars, test_size = 0.3) # 30% test data

# preparing the model on train data   -- AKU TRY LETAK POLYNOMIAL KAT HP
model_train = smf.ols("MPG ~ HP+I(HP*HP)+SP+VOL", data = cars_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid  = test_pred - cars_test.MPG

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) 
test_rmse


# train_data prediction
train_pred = model_train.predict(cars_train)

# train residual values 
train_resid  = train_pred - cars_train.MPG

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
