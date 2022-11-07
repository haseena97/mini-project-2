import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
training = pd.read_csv("C:\\Users\\Acer\\Downloads\\NB_Assignment\\SalaryData_Test (2)\\SalaryData_Train.csv")
test = pd.read_csv("C:\\Users\\Acer\\Downloads\\NB_Assignment\\SalaryData_Test (2)\\SalaryData_Test.csv")
training['train_test'] = 1 # tambah column baru utk identify training/test
test['train_test'] = 0
all_data = pd.concat([training,test]) # merge training dgn test file
all_data.columns

# Observe training data
training.describe().T # transpose version of describe
training.info()
training.columns
training.isnull().sum() # identify missing value
null_training = training[training.isnull().any(axis=1)] # kluar dkat variable
training.loc[training.isnull().any(axis=1)] # kluar kat console
trainingdropna() # remove missing value
# write an Assert statement to verify really no missing, unexpected 0 or negative values
#assert that there are no missing values in the dataframe
# return Nothing if the value is True, ada value kalau mmg error
assert pd.notnull(training).all().all()
training_cat = training[['workclass', 'education','maritalstatus','occupation', 'relationship','race', 'sex','native']]
for i in training_cat.columns:
    sns.barplot(data_training[i].value_counts().index,training_cat[i].value_counts()).set_title(i)
    plt.show()
    
    
all_data.isnull().sum() # identify missing value

#create all categorical variables that we did above apply for both training and test sets = alldata
alldata_cat = all_data[['workclass', 'education','maritalstatus','occupation', 'relationship','race', 'sex','native','Salary']]
label_encoders = {}
for column in alldata_cat:
    label_encoders[column] = LabelEncoder()
    all_data[column] = label_encoders[column].fit_transform(all_data[column]) 

#Split to train test again lepas dah selaraskan semua
X_train = all_data[all_data.train_test == 1].drop(['Salary','train_test'], axis =1)
X_test = all_data[all_data.train_test == 0].drop(['Salary','train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Salary
y_train.shape
y_test = all_data[all_data.train_test==0].Salary
# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
pd.crosstab(y_test, pred, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(y_test, pred))

# error on train data
pred_train = classifier.predict(X_train)
pd.crosstab(y_train, pred_train, rownames=['Actual'],colnames= ['Predictions']) 
print(accuracy_score(y_train, pred_train))
