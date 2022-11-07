import pandas as pd
# import numpy as np

#Importing Data
claimants1 = pd.read_csv("C:\\Users\Acer\Downloads\Logistic Regression\claimants\claimants.csv", sep=",")

#removing CASENUM column
claimants1 = claimants1.drop('CASENUM', axis=1)
claimants1.head(11) # 11 value pertama

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(claimants1, test_size = 0.34) # 30% test data

# Model building 
import statsmodels.formula.api as sm
logit_model = sm.logit('ATTORNEY ~ CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT', data = train_data).fit()

#summary
logit_model.summary()

## Evaluation of the model
predict_test = logit_model.predict(pd.DataFrame(test_data[['CLMAGE','LOSS','CLMINSUR','CLMSEX','SEATBELT']]))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# tgk confusion matrix
cnf_test_matrix = confusion_matrix(test_data['ATTORNEY'], predict_test > 0.5 )
cnf_test_matrix

print(accuracy_score(test_data.ATTORNEY, predict_test > 0.5))

## Error on train data
predict_train = logit_model.predict(pd.DataFrame(train_data[['CLMAGE','LOSS','CLMINSUR','CLMSEX','SEATBELT']]))

cnf_train_matrix = confusion_matrix(train_data['ATTORNEY'], predict_train > 0.5 )
cnf_train_matrix

print(accuracy_score(train_data.ATTORNEY, predict_train > 0.5))
