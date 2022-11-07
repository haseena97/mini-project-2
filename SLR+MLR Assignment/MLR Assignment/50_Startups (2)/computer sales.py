import pandas as pd
# 'utf-8' codec can't decode byte 0xa0 in position 679:
house = pd.read_csv("C:\\Users\\Acer\\Downloads\\50_Startups (2)\\50_Startups.csv",encoding='latin1')
# add header to pandas dataframe
df = pd.read_csv("path/to/file.txt", sep='\t')
headers =  ['CRIM',"ZN", "INDUS", "CHAS", "NOX", 'RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO','B','LSTAT', 'MEDV']
house.columns = ['research', 'admin','marketing','state','profit']
# Multilinear Regression

house.describe().columns
house2.Quarterly_Tax.describe()
house2.Quarterly_Tax.mode()
house2.drop(columns=['norm_KM'],inplace=True) # INPLACE tukar terus kat dataframe
house2.head
house2 = house[['Price', 'Age_08_04','KM', 'HP','cc', 'Doors','Gears', 'Quarterly_Tax', 'Weight']]
# check missing values
house2.isnull().sum()
# EDA
# Scatter plot between the variables along with histograms
import seaborn as sns
import matplotlib.pyplot as plt
# tgk distribution utk dependant variable
sns.distplot(house.research)
for i in house.columns:
    sns.distplot(house[i])
    plt.title(i)
    plt.show()  
sns.distplot(house.trend)
sns.countplot(x="state", data=house)

sns.boxplot(house.research)   
house.describe()
# look at numeric and categorical values separately 
house_num = house[['Price']]
house_cat = house[['cc','Doors','Gears']]       
import numpy as np
for i in house.columns:
    plt.hist(house[i])
    plt.title(i)
    plt.show()  
# sbb ada outlier sampai 16000 so replace number tu
house2["cc"] = np.where(house2["cc"] >4000, 1600,house2['cc'])
house.speed = house.speed.fillna(house.speed.median())   
plt.hist(house.speed)       
# Correlation matrix 
sns.pairplot(house.iloc[:,:]) 
house.corr()
# variable byk sgt so better tgk heatmap yg ada correlation matrix
correlation_matrix = house.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
# label encoder
#origin = [“USA”, “EU”, “EU”, “ASIA”,”USA”, “EU”, “EU”, “ASIA”, “ASIA”, “USA”]
origin_series = pd.Series(house.state)
cat_house = origin_series.astype('category')
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
house_encoded = lb_make.fit_transform(cat_house)
# nak masukkan terus dalam column training data
house['statecoded'] = house_encoded

# binary encoding
house['cd'] = house['cd'].apply(lambda x: 1 if x == 'yes' else 0)
house['multi'] = house['multi'].apply(lambda x: 1 if x == 'yes' else 0)
house['premium'] = house['premium'].apply(lambda x: 1 if x == 'yes' else 0)
house_cat = house[['cd','multi','premium']]   
correlation_matrix = house_cat.corr().round(2)
import numpy as np
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

corrmat = house.corr()
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corrmat, vmax=.8, square=True);
# Heatmap yg tunjuk correlation value
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'profit')['profit'].index
cm = np.corrcoef(house[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot correlation matrix utk selected columns
sns. regplot(x=house['Age_08_04'], y=house['Price'])
sns.catplot(x="state", y="profit", data=house)
# sebab x nak log zero
house['norm_reserach'] = np.log(house.research+1)
house['norm_marketing'] = np.log(house.marketing+1)
house['norm_research'].hist()
house['norm_marketing'].hist()
# salah spelling column name
house.rename(columns={'norm_reserach': 'norm_research'}, inplace=True)
house['norm_research'] = np.log(house.research)
house['norm_research'].hist()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('profit ~ research+admin+marketing', data=house2).fit() # simple regression model

# Summary
ml1.summary()
# check outlier
# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=ml1.get_influence().cooks_distance
c
# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(house)),np.round(c,5))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()
# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)
# identify argmax tu kat row mana
house[house.index.isin([49])] 
# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
house2=house.drop(house.index[[49]],axis=0).reset_index(drop=True)
house2
# calculating VIF's values of independent variables
rsq_price = smf.ols('price ~ speed+hd+ram+screen+cd+multi+premium+ads+trend', data=house).fit().rsquared  
vif_price = 1/(1-rsq_price) 

rsq_sp = smf.ols('speed ~ price+hd+ram+screen+cd+multi+premium+ads+trend', data=house).fit().rsquared  
vif_sp = 1/(1-rsq_sp)

rsq_hd = smf.ols('hd ~ speed+price+ram+screen+cd+multi+premium+ads+trend', data=house).fit().rsquared  
vif_hd = 1/(1-rsq_hd) 

rsq_ram = smf.ols('ram ~ speed+hd+price+screen+cd+multi+premium+ads+trend', data=house).fit().rsquared  
vif_ram = 1/(1-rsq_ram) 

rsq_scr = smf.ols('screen ~ speed+hd+ram+price+cd+multi+premium+ads+trend', data=house).fit().rsquared  
vif_scr = 1/(1-rsq_scr) 

rsq_cd = smf.ols('cd ~ speed+hd+ram+screen+price+multi+premium+ads+trend', data=house).fit().rsquared  
vif_cd = 1/(1-rsq_cd) 

rsq_mul = smf.ols('multi ~ speed+hd+ram+screen+price+price+premium+ads+trend', data=house).fit().rsquared  
vif_mul = 1/(1-rsq_mul) 

rsq_premi = smf.ols('premium ~ speed+hd+ram+screen+price+multi+price+ads+trend', data=house).fit().rsquared  
vif_premi = 1/(1-rsq_premi) 

rsq_ads = smf.ols('ads ~ speed+hd+ram+screen+price+multi+premium+price+trend', data=house).fit().rsquared  
vif_ads = 1/(1-rsq_ads) 

rsq_trend = smf.ols('trend ~ speed+hd+ram+screen+price+multi+premium+ads+price', data=house).fit().rsquared  
vif_trend = 1/(1-rsq_trend) 
# Storing vif values in a data frame
d1 = {'Variables':['price', 'speed', 'hd', 'ram', 'screen', 'cd', 'multi', 'premium',
       'ads', 'trend'],'VIF':[vif_price,vif_sp,vif_hd,vif_ram,vif_scr,vif_cd,vif_mul,vif_premi,vif_ads,vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
#kalau nak write dalam excel file (nak header=true,kalau x nak header=None,kalau x nak index=False)sep=separator
Vif_frame.to_csv('C:\\Users\\Acer\\Downloads\\Cars\\test2.csv',header=True,index=False,sep=',',mode='w')
#kalau nak write dalam txt file (nak header=true,kalau x nak header=None,kalau x nak index=False),\t bagi sengkang
Vif_frame.to_csv('C:\\Users\\Acer\\Downloads\\Cars\\test2.txt',header=True,index=False,sep='\t',mode='w')

# scatter plot
import numpy as np
plt.scatter(x=(house['Age_08_04']),y=house['Price'],color='blue')
plt.scatter(x=(house['RM']),y=house['MEDV'],color='red')
plt.scatter(x=np.log(house['TAX']),y=house['MEDV'],color='yellow')
plt.scatter(x=(house['PTRATIO']),y=house['MEDV'],color='blue')
plt.scatter(x=(house['LSTAT']),y=house['MEDV'],color='black')
plt.legend(loc='upper left')
plt.show()

# final model (letak log dekat TAX & polynomial LSTAT)
final_ml = smf.ols('Price ~ Age_08_04+I(Age_08_04*Age_08_04)+KM+HP+Quarterly_Tax+cc+Doors+Gears', data=house2).fit() # simple regression model
final_ml.summary()
model_train = smf.ols('profit ~ research+admin+marketing', data=house2).fit() # simple regression model

model_train.rsquared

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
house_train, house_test  = train_test_split(house2, test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04+I(Age_08_04*Age_08_04)+KM+HP+Quarterly_Tax+cc+Doors+Gears+Weight', data = house_train).fit()
model_train.summary()
# prediction on test data set 
test_pred = model_train.predict(house_test)
test_pred.to_csv("C:\\Users\\Acer\\Downloads\\50_Startups (2)\\test2.txt",header=True,index=False,sep='\t',mode='w')

# test residual values 
test_resid  = test_pred - house_test.profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) 
test_rmse


# train_data prediction
train_pred = model_train.predict(house_train)

# train residual values 
train_resid  = train_pred - house_train.profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
