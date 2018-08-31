
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")

TrainTestDataDir = '/home/songyue/homeCredit/HomeCreditDefaultRisk/Data/TrainTestData'
modelDir = '/home/songyue/homeCredit/HomeCreditDefaultRisk/GBDT'

def loadData(DataName):
    DataPath = os.path.join(TrainTestDataDir, DataName)
    if not os.path.exists(DataPath):
        print('%s does not exist!' % DataPath)
        return
    OriginData = pd.read_csv(DataPath, index_col=0)
    OriginData = OriginData.sample(frac=1)  # 打乱顺序后返回
    return OriginData


def NormalData(TrainData, TestData):
    # 对一些列的均值大于100的进行归一化处理
    AllData = TrainData.append(TestData)
    for col in AllData.columns:
        if abs(AllData[col].mean())>1:
            scaler = StandardScaler().fit(np.atleast_2d(AllData[col]).T)
            TrainData[col] = scaler.transform(np.atleast_2d(TrainData[col]).T)
    return TrainData
ValidData = loadData('validation.csv')
ValidID = ValidData.SK_ID_CURR.values
# ValidData.drop('SK_ID_CURR', axis=1, inplace=True)
TrainData = loadData('train.csv')
TrainData.reset_index(drop=True, inplace=True)
TrainDataPos = TrainData[TrainData.TARGET==1]
TrainDataNeg = TrainData[TrainData.TARGET==0]
# TrainDataTest = TrainDataPos.append(TrainDataNeg.iloc[:len(TrainDataNeg)//10,:])
# 代价敏感学习
TrainDataTest = TrainDataNeg.append(TrainDataPos.iloc[:,:])
for i in range(9):
    TrainDataTest.append(TrainDataPos.iloc[:,:])

TrainDataTest.reset_index(drop=True,inplace=True)
print(len(TrainDataTest))
# TrainDataTest.head()
TrainDataTest.drop('SK_ID_CURR', axis=1, inplace=True)
ValidData.drop('SK_ID_CURR', axis=1, inplace=True)
def NormalData2(Data):
    for col in Data.columns:
        if abs(Data[col].mean())>1:
            scaler = StandardScaler().fit(np.atleast_2d(Data[col]).T)
            Data[col] = scaler.transform(np.atleast_2d(Data[col]).T)
    return Data
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0:'Missing Values',1:'% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1]!=0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

gbdt=GradientBoostingClassifier(max_depth=8,max_features=20,n_estimators=80)
ModelDir = 'models'
ModelName = 'train1'
TrainTargetTest = TrainDataTest.TARGET
ValTargetTest = ValidData.TARGET
# print(TrainTargetTest)
print(len(TrainTargetTest))
print(len(ValTargetTest))
TrainDatatest = NormalData2(TrainDataTest)
ValidDatatest = NormalData2(ValidData)
TrainDatatest.head()
ValidDatatest.head()
X_train = TrainDatatest.drop('TARGET', axis=1).values
# X_train = X_train.values
y_train = TrainTargetTest.values
X_val = ValidDatatest.drop('TARGET', axis=1).values
y_val = ValTargetTest.values
# gbdt.fit(X_train,y_train)
# pred = gbdt.predict(X_val)
# pd.crosstab(y_val,pred)
# m_dir = os.path.join(modelDir,ModelDir)
# joblib.dump(gbdt,os.path.join(m_dir,'gbdt4.m'))
# print('acc: ',accuracy_score(y_val,pred))
# print('roc: ',roc_auc_score(y_val,pred))
# print('\n')


# In[ ]:

#from sklearn.grid_search import GridSearchCV

#param_test1 = {'n_estimators':np.arange(20,81,10)}
#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                 # min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
 #                      param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
#gsearch1.fit(X_train,y_train)
#print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
# from sklearn.grid_search import GridSearchCV

# param_test1 = {'n_estimators':np.arange(20,81,10)}
# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
#                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
# gsearch1.fit(X_train,y_train)
# gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
# param_test2 = {'n_estimators':np.arange(90,120,10)}
# gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
#                        param_grid = param_test2, scoring='roc_auc',iid=False,cv=5)
# gsearch2.fit(X_train,y_train)
# gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
# param_test3 = {'n_estimators':np.arange(100,151,10),
#                'max_features':np.arange(7,25,2),
#                 'max_depth':np.arange(3,14,2)}
# gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                   min_samples_leaf=20,subsample=0.8,random_state=10), 
#                        param_grid = param_test3, scoring='roc_auc',iid=False,cv=5)
#gsearch3.fit(X_train,y_train)
#gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
param_test3 = {'max_depth':np.arange(5,20,2)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,n_estimators=150,
                                   min_samples_leaf=20,subsample=0.8,random_state=10), 
                       param_grid = param_test3, scoring='roc_auc',iid=False,
                        n_jobs=-1,cv=5)
gsearch3.fit(X_train,y_train)
print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)


# In[ ]:

# # 训练10个模型

# n = 10

# for i in range(10):
#     batch_size = len(TrainDataNeg)//10
#     if i == 9:
#         trainData = TrainDataPos.append(TrainDataNeg.iloc[9*batch_size:,:])
#     else:
#         trainData = TrainDataPos.append(TrainDataNeg.iloc[i*batch_size:(i+1)*batch_size,:])
#     trainData.reset_index(drop=True,inplace=True)
#     trainData.drop('SK_ID_CURR', axis=1, inplace=True)
#     gbdt=GradientBoostingClassifier(max_depth=gsearch3.best_params_['max_depth'],
#                                     max_features=gsearch3.best_params_['max_features'],
#                                     n_estimators=gsearch3.best_params_['n_estimators'])
#     ModelDir = 'models'
#     ModelName = 'train1'
#     trainTarget = trainData.TARGET
#     trainData = NormalData2(trainData)
#     X_train = trainData.drop('TARGET', axis=1).values
#     # X_train = X_train.values
#     y_train = trainData.values
# #     X_val = ValidDatatest.drop('TARGET', axis=1).values
# #     y_val = ValTargetTest.values
#     gbdt.fit(X_train,y_train)
#     pred = gbdt.predict(X_val)
#     pd.crosstab(y_val,pred)
#     m_dir = os.path.join(modelDir,ModelDir)
#     model_name = 'gbdt_%s.m' % str(i+1)
#     joblib.dump(gbdt,os.path.join(m_dir,model_name))
#     print('acc: ',accuracy_score(y_val,pred))
#     print('roc: ',roc_auc_score(y_val,pred))
#     print('\n')

