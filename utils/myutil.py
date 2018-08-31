import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import re

# 加载已经做好的columns描述文件
OriginDataDir = 'Data/OriginData/'
descPath = os.path.join(OriginDataDir, 'HomeCredit_columns_description.csv')
desc = pd.read_csv(descPath, encoding='ISO-8859-1')


def getDescOfColumns(DataPath):
    # 获取每一列的描述文字
    return desc


def showPNGRelativeCSV():
    PNGpath = os.path.join(OriginDataDir, 'home_credit.png')
    img = mpimg.imread(PNGpath)
    plt.imshow(img)
    plt.show()


def DescofCol(csvfile, colname):
    csvfilenew = csvfile.replace('train', '*')
    csvfilenew = csvfilenew.replace('test', '*')
    index = csvfilenew + colname
    return desc[index][0]
