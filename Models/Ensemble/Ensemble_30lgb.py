import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from skopt.space import Real, Categorical, Integer
import xgboost as xgb
from xgboost import XGBClassifier
import os
import gc
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import lightgbm as lgb
from sklearn.svm import LinearSVC, SVC
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random
import warnings
warnings.simplefilter('ignore')
TrainDataDir = 'data'
ResultSaveDir = 'result_ensemble'

# 设置的超参数
LgbmNum = 20
TrainDataName_withCate = 'train_mini5.csv'
TrainDataName_withoutCate = 'train_mini5.csv'

# 设置lgbm的参数，有两种，一种是有category特征，一种是没有的
model_config_withCate = {  #'random_search_runs': 0,
                      'device': 'cpu', # gpu cpu
                      'num_threads':-1,
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': 'auc',
                      'learning_rate': 0.015,
                      'max_bin': 300,
                      'max_depth': -1,
                      'num_leaves': 30,
                      'min_child_samples': 40,
                      'subsample': 1,
                      'subsample_freq': 1,
                      'colsample_bytree': 0.03,
                      'min_gain_to_split': 2,
                      'reg_lambda': 110,
                      'reg_alpha': 0.0,
                      'scale_pos_weight': 1,  
                      'is_unbalance': False}
model_config_withoutCate = {  #'random_search_runs': 0,
                      'device': 'cpu', # gpu cpu
                      'num_threads':-1,
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': 'auc',
                      'learning_rate': 0.015,
                      'max_bin': 300,
                      'max_depth': -1,
                      'num_leaves': 30,
                      'min_child_samples': 40,
                      'subsample': 1,
                      'subsample_freq': 1,
                      'colsample_bytree': 0.03,
                      'min_gain_to_split': 2,
                      'reg_lambda': 110,
                      'reg_alpha': 0.0,
                      'scale_pos_weight': 1,  
                      'is_unbalance': False}
# 获取特征重要度的函数


def get_importances(feature_importance_df_):
    importances = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(by="importance", ascending=False)
    return importances


def loadData(Dir, DataName, index_col_flag=False):
    DataPath = os.path.join(Dir, DataName)
    if not os.path.exists(DataPath):
        print('%s does not exist!' % DataPath)
        return
    if index_col_flag:
        OriginData = pd.read_csv(DataPath, index_col=0)
    else:
        OriginData = pd.read_csv(DataPath)
    # OriginData = OriginData.sample(frac=1)  # 打乱顺序后返回
    return OriginData


def RandomDisturbance(model_config):
    model_config_random = {
        'learning_rate': model_config['learning_rate'] + random.uniform(-0.001, 0.001),
        'num_leaves': model_config['num_leaves'] + random.randint(-1, 1),
        'subsample': model_config['subsample'] - random.uniform(0, 0.01),
        'colsample_bytree': model_config['colsample_bytree'] + random.uniform(-0.001, 0.001),
        'reg_lambda': model_config['reg_lambda'] + random.randint(-1, 1),
    }
    return model_config_random


def saveData(IsCategory, NumofModel, importance, submission, oof_train):
    if IsCategory:
        ResultSaveSubDir = os.path.join(ResultSaveDir, 'WithCategory')
        if not os.path.exists(ResultSaveSubDir):
            os.mkdir(ResultSaveSubDir)
    else:
        ResultSaveSubDir = os.path.join(ResultSaveDir, 'WithoutCategory')
        if not os.path.exists(ResultSaveSubDir):
            os.mkdir(ResultSaveSubDir)
    # 保存数据
    SaveDirPath = os.path.join(ResultSaveSubDir, 'lgbmodel%d' % NumofModel)
    if not os.path.exists(SaveDirPath):
        os.mkdir(SaveDirPath)
    importance.to_csv(os.path.join(SaveDirPath, "importance.csv"))
    submission.to_csv(os.path.join(SaveDirPath, "submission.csv"))
    oof_train.to_csv(os.path.join(SaveDirPath, "oof_train.csv"))


def train(TrainDataName, IsCategory):
    gc.enable()
    print('loading data:%s'%TrainDataName)
    TrainData = loadData(TrainDataDir, TrainDataName, True)
    if IsCategory:
        model_config = model_config_withCate.copy()
        features = [x for x in TrainData.columns if x !=
                    'TARGET' and x != 'SK_ID_CURR']
        ObjColList = TrainData.select_dtypes(include='object').columns.tolist()
        for objcol in ObjColList:
            TrainData[objcol] = TrainData[objcol].astype('category')
    else:
        model_config = model_config_withoutCate.copy()
        features = [x for x in TrainData.columns if x !=
                    'TARGET' and x != 'SK_ID_CURR']
        ObjColList = TrainData.select_dtypes(include='object').columns.tolist()
        for objcol in ObjColList:
            if objcol in features:
                features.remove(objcol)
    df_test = TrainData[TrainData.TARGET.isnull()]
    df_test.reset_index(drop=True, inplace=True)
    df_train = TrainData[~TrainData.TARGET.isnull()]
    df_train.reset_index(drop=True, inplace=True)
    del TrainData
    gc.collect()
    for i in range(LgbmNum):
        t0 = time.time()
        print('------>Using Category:%s' % ('True' if IsCategory else 'False'))
        print('------>This is the %dth model...' % (i+1))
        print("------>Starting LGBM. Train shape: {}, test shape: {}".format(
            df_train.shape, df_test.shape))
        print("------>Num of Feature:", len(features))
        # Cross validation model
        num_folds = random.choice([5, 7])
        stratified = random.choice([0, 1])
        # 随机扰动参数
        model_config_in = model_config_withCate.copy(
        ) if IsCategory else model_config_withoutCate.copy()
        model_config.update(RandomDisturbance(model_config_in))
        print(model_config)
        if stratified:
            print('------>StratifiedKFold:%d...' % num_folds)
            folds = StratifiedKFold(
                n_splits=num_folds, shuffle=True, random_state=random.randint(0, 100000))
        else:
            print('------>KFold%d...' % num_folds)
            folds = KFold(n_splits=num_folds, shuffle=True,
                          random_state=random.randint(0, 100000))
        # Create arrays and dataframes to store results
        oof_preds = np.zeros(df_train.shape[0])
        sub_preds = np.zeros(df_test.shape[0])
        roc_score_list = []
        feature_importance_df = pd.DataFrame()

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[features], df_train['TARGET'])):
            train_x, train_y = df_train[features].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
            valid_x, valid_y = df_train[features].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]

            data_train = lgb.Dataset(
                data=train_x, label=train_y, feature_name='auto', categorical_feature='auto')
            data_valid = lgb.Dataset(
                data=valid_x, label=valid_y, feature_name='auto', categorical_feature='auto')
            clf = lgb.train(model_config,
                            data_train,
                            valid_sets=[data_train, data_valid],
                            valid_names=['data_train', 'data_valid'],
                            num_boost_round=10000,
                            early_stopping_rounds=200,
                            verbose_eval=200,
                            feature_name='auto',
                            categorical_feature='auto')

            oof_preds[valid_idx] = clf.predict(
                valid_x, num_iteration=clf.best_iteration)
            sub_preds += clf.predict(df_test[features],
                                     num_iteration=clf.best_iteration)/num_folds
            roc_curr = roc_auc_score(valid_y, oof_preds[valid_idx])
            roc_score_list.append(roc_curr)
            print('Fold %d AUC : %.6f' % (n_fold + 1, roc_curr))
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = features
            fold_importance_df["importance"] = clf.feature_importance()
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat(
                [feature_importance_df, fold_importance_df], axis=0)
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()
        print('------>The %dth model uses %f s' % (i+1, time.time()-t0))
        print('------>mean of model:%f' % np.mean(roc_score_list))
        print('------>std of model:%f' % np.std(roc_score_list))
        # 保存数据
        submission = pd.DataFrame(
            {'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds})
        oof_train = pd.DataFrame(
            {'SK_ID_CURR': df_train['SK_ID_CURR'], 'pred': oof_preds})
        importance = get_importances(fold_importance_df)
        print('------>Save Data...')
        saveData(IsCategory, i, importance, submission, oof_train)


def getMeanSub():
    ModelCount = 0
    submission_ensemble = pd.DataFrame()
    ResultSaveSubDirWithCategory = os.path.join(ResultSaveDir, 'WithCategory')
    ResultSaveSubDirWithoutCategory = os.path.join(
        ResultSaveDir, 'WithoutCategory')
    if os.path.exists(ResultSaveSubDirWithCategory):
        print('reading WithCategory')
        for SubDir in os.listdir(ResultSaveSubDirWithCategory):
            print(SubDir)
            submission = pd.read_csv(os.path.join(
                ResultSaveSubDirWithCategory, SubDir, 'submission.csv'), index_col=0)
            if len(submission_ensemble):
                submission_ensemble['TARGET'] += submission['TARGET']
            else:
                submission_ensemble = submission.copy()
            ModelCount += 1
    if os.path.exists(ResultSaveSubDirWithoutCategory):
        print('reading WithoutCategory')
        for SubDir in os.listdir(ResultSaveSubDirWithoutCategory):
            print(SubDir)
            submission = pd.read_csv(os.path.join(
                ResultSaveSubDirWithoutCategory, SubDir, 'submission.csv'), index_col=0)
            if len(submission_ensemble):
                submission_ensemble['TARGET'] += submission['TARGET']
            else:
                submission_ensemble = submission.copy()
            ModelCount += 1
    submission_ensemble['TARGET'] /= ModelCount
    return submission_ensemble


def main():
    print('------>Start train lgb model with category...')
    train(TrainDataName_withCate, 1)
    #print('------>Start train lgb model without category...')
    #train('train_mini3.csv', 0)
    # 将预测的30个结果做一个均值
    #submission_ensemble = getMeanSub()
    #submission_ensemble.to_csv(os.path.join(
        #ResultSaveDir, 'submission_ensemble.csv'), index=False)


if __name__ == '__main__':
    main()
