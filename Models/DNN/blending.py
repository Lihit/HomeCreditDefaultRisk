import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

def LREnsemble():
    ModelName = 'RandomDNNEnsembleLR'
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
    TrainDataNew = np.array(RandomDNNForTrain(
        nets, TrainData, FeatureRandomSelection), dtype=np.float32)
    TestDataNew = np.array(RandomDNNForTrain(
        nets, TestData, FeatureRandomSelection), dtype=np.float32)
    lr = LogisticRegression()
    scores = cross_val_score(lr, TrainDataNew, TrainTarget.values, cv = 5, scoring='roc_auc')
    print(scores)
    lr.fit(TrainDataNew,TrainTarget.values)
    pred = lr.predict_proba(TestDataNew)[:,1]
    gender_submission = pd.DataFrame({'SK_ID_CURR':TestID,'TARGET':pred})
    gender_submission.to_csv('result.csv', index = False)
    

def DNNEnsemble():
    ModelName = 'RandomDNNEnsembleDNN'
    with open(os.path.join(DNNModelsdIR,'FeatureRandomSelection.json'),'r') as fp:
        FeatureRandomSelection = json.load(fp)
    # 先加载保存好的模型
    nets = [torch.load(os.path.join(DNNModelsdIR, 'randomDNN%d.pkl'%(i+1)))
            for i in range(len(FeatureRandomSelection))]
    # 把所有的训练数据都加载出来
    TrainData = loadData('AllTrain.csv')
    TrainData.reset_index(drop=True, inplace=True)
    TrainData.drop('SK_ID_CURR', axis=1, inplace=True)
    TestData = loadData('test.csv')
    TestData.drop('SK_ID_CURR', axis=1, inplace=True)
    TrainTarget = TrainData.TARGET
    TrainData.drop('TARGET', axis=1, inplace=True)
    TrainData, TestData = NormalData(TrainData, TestData)
    TrainDataNew = np.array(RandomDNNForTrain(
        nets, TrainData, FeatureRandomSelection), dtype=np.float32)
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(TrainDataNew,TrainTarget.values, test_size=0.3, random_state=711)
    # 将他们转化为torch tensor
    train_x = torch.from_numpy(X_train).type(torch.FloatTensor)
    train_y = torch.from_numpy(y_train).type(torch.FloatTensor)
    val_x = Variable(torch.from_numpy(X_val).type(
        torch.FloatTensor), volatile=True).cuda()
    val_y = Variable(torch.from_numpy(y_val).type(
        torch.FloatTensor), volatile=True).cuda()
    ##DNN的一些超参数
    BATCH_SIZE = 4096
    INPUT_SIZE = train_x.size(1)
    OUTPUT_SIZE = 1
    LR = 0.0007
    EPOCHES = 100
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    MyNet = torch.nn.Sequential(
        torch.nn.Linear(INPUT_SIZE,32),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(32,OUTPUT_SIZE),
        torch.nn.Sigmoid())
    # 使用GPU训练
    MyNet.cuda()
    print(MyNet)
    optimizer = torch.optim.Adam(MyNet.parameters(), lr=LR)
    loss_func = torch.nn.BCELoss().cuda()

    # 开始训练
    trainlosses = []
    trainrosscores = []
    vallosses = []
    valrosscores = []
    for epoch in range(EPOCHES):
        trainlossTmp = []
        trainRocTmp = []
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = Variable(b_x).type(torch.FloatTensor).cuda(), Variable(
                b_y).type(torch.FloatTensor).cuda()
            pred = MyNet(b_x)
            trainloss = loss_func(pred, b_y)
            optimizer.zero_grad()
            trainloss.backward()
            optimizer.step()
            trainRocTmp.append(roc_auc_score(
                b_y.data.cpu().numpy().flatten(), pred.data.cpu().numpy().flatten()))
            trainlossTmp.append(trainloss.data[0])
        trainrosscores.append(np.mean(trainRocTmp))
        trainlosses.append(np.mean(trainlossTmp))
        print('Epoch: ', epoch)
        MyNet.eval()
        pred = MyNet(val_x)
        valloss = loss_func(pred, val_y)
        vallosses.append(valloss.data[0])
        valroc = roc_auc_score(val_y.data.cpu().numpy().flatten(),
                               pred.data.cpu().numpy().flatten())
        valrosscores.append(valroc)
        print('Train roc score:', np.mean(trainRocTmp))
        print('Train loss:', np.mean(trainlossTmp))
        print('Validation roc score:', valroc)
        print('Validation loss:', valloss.data[0])
        MyNet.train()
    # 最后保存模型
    torch.save(MyNet, os.path.join(DNNModelsdIR, ModelName + '.pkl'))
    # 画出loss的曲线图
    plt.figure()
    plt.title('Loss')
    plt.plot(range(len(trainlosses)), trainlosses, label='TrainLoss')
    plt.plot(range(len(vallosses)), vallosses, label='ValidationLoss')
    plt.legend()
    plt.pause(0.1)
    plt.savefig(os.path.join(FifSaveDir,'loss%s.jpg'%ModelName))
    plt.figure()
    plt.title('ROC')
    plt.plot(range(len(trainrosscores)), trainrosscores, label='TrainROC')
    plt.plot(range(len(valrosscores)), valrosscores, label='ValidationROC')
    plt.pause(0.1)
    plt.savefig(os.path.join(FifSaveDir,'roc%s.jpg'%ModelName))
    
if __name__ == '__main__':
    DNNEnsemble()