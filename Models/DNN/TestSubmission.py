import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import json
import matplotlib.pyplot as plt
import warnings
plt.switch_backend('agg')
warnings.simplefilter("ignore")

TrainTestDataDir = '/home/songyue/homeCredit/HomeCreditDefaultRisk/Data/TrainTestData'
DNNModelsdIR = '/home/songyue/homeCredit/HomeCreditDefaultRisk/DNN/models'
FifSaveDir = '/home/songyue/homeCredit/HomeCreditDefaultRisk/DNN/result/'
SubTestDir = '/home/songyue/homeCredit/HomeCreditDefaultRisk/DNN/SubmissionResult/'


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
        if abs(AllData[col].mean()) > 1:
            scaler = StandardScaler().fit(np.atleast_2d(AllData[col]).T)
            TrainData[col] = scaler.transform(np.atleast_2d(TrainData[col]).T)
            TestData[col] = scaler.transform(np.atleast_2d(TestData[col]).T)
    return TrainData, TestData


def DNNOutputForTrain(nets, TrainData):
    TrainDataNew = []
    BatchSize = 4096
    beginIndex = 0
    while beginIndex<=len(TrainData):
        SigleTrain = []
        x = Variable(torch.from_numpy(TrainData.iloc[beginIndex:min(beginIndex+BatchSize,len(TrainData)), :].values).type(
            torch.FloatTensor), volatile=True).cuda()
        for net in nets:
            net.eval()
            pred = net(x)
            SigleTrain.append(pred.data.cpu().numpy())
        TrainDataNew.append(np.hstack(SigleTrain))
        beginIndex += BatchSize
    return np.vstack(TrainDataNew)

def RandomDNNForTrain(nets, TrainData, FeatureRandomSelection):
    TrainDataNew = []
    BatchSize = 4096
    beginIndex = 0
    while beginIndex<=len(TrainData):
        SigleTrain = []
        for i,net in enumerate(nets):
            x = Variable(torch.from_numpy(TrainData.iloc[beginIndex:min(beginIndex+BatchSize,len(TrainData)), FeatureRandomSelection[i]].values).type(
            torch.FloatTensor), volatile=True).cuda()
            net.eval()
            pred = net(x)
            SigleTrain.append(pred.data.cpu().numpy())
        TrainDataNew.append(np.hstack(SigleTrain))
        beginIndex += BatchSize
    return np.vstack(TrainDataNew)

def ensemble():
    with open(os.path.join(DNNModelsdIR,'FeatureRandomSelection.json'),'r') as fp:
        FeatureRandomSelection = json.load(fp)
    # 先加载保存好的模型
    nets = [torch.load(os.path.join(DNNModelsdIR, 'randomDNN%d.pkl'%(i+1)))
            for i in range(len(FeatureRandomSelection))]
    print('Num of Models:',len(nets))
    # 把所有的训练数据都加载出来
    TrainData = loadData('AllTrain.csv')
    TrainData.reset_index(drop=True, inplace=True)
    TrainData.drop('SK_ID_CURR', axis=1, inplace=True)
    TestData = loadData('test.csv')
    TestID = TestData.SK_ID_CURR.values
    TestData.drop('SK_ID_CURR', axis=1, inplace=True)
    TrainTarget = TrainData.TARGET
    TrainData.drop('TARGET', axis=1, inplace=True)
    TrainData, TestData = NormalData(TrainData, TestData)
    TestDataNew = np.array(RandomDNNForTrain(
        nets, TestData, FeatureRandomSelection), dtype=np.float32)
    MyEnsemble = torch.load(os.path.join(DNNModelsdIR, 'RandomDNNEnsembleDNN.pkl'))
    test_x = Variable(torch.from_numpy(TestDataNew).type(
        torch.FloatTensor), volatile=True).cuda()
    MyEnsemble.cuda()
    MyEnsemble.eval()
    pred = MyEnsemble(test_x)
    gender_submission = pd.DataFrame({'SK_ID_CURR':TestID,'TARGET':pred.data.cpu().numpy().flatten()})
    gender_submission.to_csv('result_luck.csv', index = False)

def singleDNN():
    MyNet = torch.load(os.path.join(DNNModelsdIR, 'DNN_WEIGHT_0762_online.pkl'))
    # 把所有的训练数据都加载出来
    #TrainData = loadData('AllTrainSMOTE.csv')
    #TrainData.reset_index(drop=True, inplace=True)
    TestData = loadData('test.csv')
    TestID = TestData.SK_ID_CURR.values
    TestData.drop('SK_ID_CURR', axis=1, inplace=True)
    print(TestData.shape)
    #TrainData.drop('TARGET', axis=1, inplace=True)
    #_,TestData = NormalData(TrainData, TestData)
    test_x = Variable(torch.from_numpy(TestData.values).type(
        torch.FloatTensor), volatile=True).cuda()
    MyNet.eval()
    pred = MyNet(test_x)
    gender_submission = pd.DataFrame({'SK_ID_CURR':TestID,'TARGET':pred.data.cpu().numpy().flatten()})
    print('Gen test submission....')
    gender_submission.to_csv(os.path.join(SubTestDir,'DNN_WEIGHT_0762_online.csv'), index = False)

if __name__ == '__main__':
    singleDNN()