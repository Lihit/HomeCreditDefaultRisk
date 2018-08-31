import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt
import warnings
plt.switch_backend('agg')

warnings.simplefilter("ignore")

TrainTestDataDir = '/home/songyue/homeCredit/HomeCreditDefaultRisk/Data/TrainTestData'

def GenMyDNN(InputSize=220, OutputSize=1, DropPro=0.5):
    DNNnet = torch.nn.Sequential(
        torch.nn.Linear(InputSize, 512),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.7),
        torch.nn.Linear(256, 128),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.Dropout(0.3),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, OutputSize),
        torch.nn.Sigmoid()
    )
    return DNNnet


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


def train():
    ModelDir = 'models'
    ModelName = 'train10'
    TrainDataName = ModelName + '.csv'
    TestDataName = 'test.csv'
    TrainData = loadData(TrainDataName)
    TestData = loadData(TestDataName)
    # 更改训练样本的index
    TrainData.set_index('SK_ID_CURR', inplace=True)
    TestData.set_index('SK_ID_CURR', inplace=True)
    TrainTarget = TrainData.TARGET
    TrainData.drop('TARGET', axis=1, inplace=True)
    TrainData = NormalData(TrainData, TestData)
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(TrainData.values,
                                                      TrainTarget.values, test_size=0.1,
                                                      random_state=711)
    # 将他们转化为torch tensor
    train_x = torch.from_numpy(X_train).type(torch.FloatTensor)
    train_y = torch.from_numpy(y_train).type(torch.FloatTensor)
    val_x = Variable(torch.from_numpy(X_val).type(torch.FloatTensor), volatile=True).cuda()
    val_y = Variable(torch.from_numpy(y_val).type(torch.FloatTensor), volatile=True).cuda()

    ##DNN的一些超参数
    BATCH_SIZE = 1024
    INPUT_SIZE = train_x.size(1)
    OUTPUT_SIZE = 1
    LR = 0.0005
    EPOCHES = 20

    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    MyNet = GenMyDNN(InputSize=INPUT_SIZE, OutputSize=OUTPUT_SIZE)
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
            b_x, b_y = Variable(b_x).type(torch.FloatTensor).cuda(), Variable(b_y).type(torch.FloatTensor).cuda()
            pred = MyNet(b_x)
            trainloss = loss_func(pred, b_y)
            optimizer.zero_grad()
            trainloss.backward()
            optimizer.step()
            trainRocTmp.append(roc_auc_score(b_y.data.cpu().numpy().flatten(), pred.data.cpu().numpy().flatten()))
            trainlossTmp.append(trainloss.data[0])
        trainrosscores.append(np.mean(trainRocTmp))
        trainlosses.append(np.mean(trainlossTmp))
        print('Epoch: ', epoch)
        MyNet.eval()
        pred = MyNet(val_x)
        valloss = loss_func(pred, val_y)
        vallosses.append(valloss.data[0])
        valroc = roc_auc_score(val_y.data.cpu().numpy().flatten(), pred.data.cpu().numpy().flatten())
        valrosscores.append(valroc)
        print('Train roc score:', np.mean(trainRocTmp))
        print('Train loss:', np.mean(trainlossTmp))
        print('Validation roc score:', valroc)
        print('Validation loss:', valloss.data[0])
        MyNet.train()
    # 最后保存模型
    torch.save(MyNet, os.path.join(ModelDir, ModelName + '.pkl'))
    # 画出loss的曲线图
    plt.figure()
    plt.title('Loss')
    plt.plot(range(len(trainlosses)), trainlosses, label='TrainLoss')
    plt.plot(range(len(vallosses)), vallosses, label='ValidationLoss')
    plt.legend()
    plt.pause(0.1)
    plt.savefig('/home/songyue/homeCredit/HomeCreditDefaultRisk/DNN/loss10.jpg')
    plt.figure()
    plt.title('ROC')
    plt.plot(range(len(trainrosscores)), trainrosscores, label='TrainROC')
    plt.plot(range(len(valrosscores)), valrosscores, label='ValidationROC')
    plt.pause(0.1)
    plt.savefig('/home/songyue/homeCredit/HomeCreditDefaultRisk/DNN/roc10.jpg')


if __name__ == '__main__':
    train()