import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib
import random
import datetime
from numpy.random import seed
from imblearn.over_sampling import SMOTE


def dataSplit(dataset,m,n):
    dataset.dropna(inplace=True)
    trainData = dataset[:m]
    testData = dataset[n:]
    y_train = trainData.iloc[:,6]
    X_train = trainData.iloc[:,7:]
    
    X_train,y_train = balanceData(X_train,y_train)
    y_test = testData.iloc[:,6]
    X_test = testData.iloc[:,7:]
    return X_train,y_train,X_test,y_test  


def balanceData(X,y):
    if(sum(y==1)/sum(y==0)<=0.85) or (sum(y==0)/sum(y==1)<=0.85):
        sm = SMOTE(random_state=2)
        X, y = sm.fit_sample(X, y)    
    return pd.DataFrame(X),pd.Series(y)

