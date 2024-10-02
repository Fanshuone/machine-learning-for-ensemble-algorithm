#

#/usr/bin/env python

# -*- codeing:utf-8 -*-

"""自动模型融合模块
"""


__author__ = '九天Hector'

__version__= '0.1'

#######################################################
## Part 1.相关依赖库
# 基础数据科学运算库
import numpy as np
import pandas as pd

# 可视化库
import seaborn as sns
import matplotlib.pyplot as plt

# 时间模块
import time

import warnings
warnings.filterwarnings('ignore')

# sklearn库
# 数据预处理
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# 实用函数
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

# 常用评估器
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# 网格搜索
from sklearn.model_selection import GridSearchCV

# 自定义评估器支持模块
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# 自定义模块
from telcoFunc import *
# 导入特征衍生模块
import features_creation as fc
from features_creation import *

# re模块相关
import inspect, re

# 其他模块
from tqdm import tqdm
import gc
from joblib import dump, load
from hyperopt import hp, fmin, tpe, Trials
from numpy.random import RandomState

#######################################################
## Part 2.基础辅助函数和类

class VotingClassifier_threshold(BaseEstimator, ClassifierMixin, TransformerMixin):
    
    def __init__(self, estimators, voting='hard', weights=None, thr=0.5):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.thr = thr
        
    def fit(self, X, y):
        VC = VotingClassifier(estimators = self.estimators, 
                              voting = self.voting, 
                              weights = self.weights)
        
        VC.fit(X, y)
        self.clf = VC
        
        return self
        
    def predict_proba(self, X):
        if self.voting == 'soft':
            res_proba = self.clf.predict_proba(X)
        else:
            res_proba = None
        return res_proba
    
    def predict(self, X):
        if self.voting == 'soft':
            res = (self.clf.predict_proba(X)[:, 1] >= self.thr) * 1
        else:
            res = self.clf.predict(X)
        return res
    
    def score(self, X, y):
        acc = accuracy_score(self.predict(X), y)
        return acc

class logit_threshold(BaseEstimator, ClassifierMixin, TransformerMixin):
    
    def __init__(self, penalty='l2', C=1.0, max_iter=1e8, solver='lbfgs', l1_ratio=None, class_weight=None, thr=0.5):
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.thr = thr
        self.class_weight = class_weight
        
    def fit(self, X, y):
        clf = LogisticRegression(penalty = self.penalty, 
                                 C = self.C, 
                                 solver = self.solver, 
                                 l1_ratio = self.l1_ratio,
                                 class_weight=self.class_weight, 
                                 max_iter=self.max_iter, 
                                 random_state=12)
        clf.fit(X, y)
        self.coef_ = clf.coef_
        self.clf = clf
        self.classes_ = pd.Series(y).unique()
        return self
        
    def predict_proba(self, X):
        res_proba = self.clf.predict_proba(X)
        return res_proba
    
    def predict(self, X):
        res = (self.clf.predict_proba(X)[:, 1]>=self.thr) * 1
        return res
    
def train_cross(X_train, y_train, X_test, estimators, test_size=0.2, n_splits=5, random_state=12, blending=False):
    """
    Stacking融合过程一级学习器交叉训练函数
    
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param X_test: 测试集特征
    :param estimators: 一级学习器，由(名称,评估器)组成的列表
    :param n_splits: 交叉训练折数
    :param test_size: blending过程留出集占比
    :param random_state: 随机数种子
    :param blending: 是否进行blending融合
    
    :return：交叉训练后创建oof训练数据和测试集平均预测结果，同时包含特征和标签，标签在最后一列
    """    
    # 创建一级评估器输出的训练集预测结果和测试集预测结果数据集
    if blending == True:
        X, X1, y, y1 = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
        m = X1.shape[0]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X1 = X1.reset_index(drop=True)
        y1 = y1.reset_index(drop=True)
    else:
        m = X_train.shape[0]
        X = X_train.reset_index(drop=True)
        y = y_train.reset_index(drop=True)
    
    n = len(estimators)
    m_test = X_test.shape[0]
    
    columns = []
    for estimator in estimators:
        columns.append(estimator[0] + '_oof')
    
    train_oof = pd.DataFrame(np.zeros((m, n)), columns=columns)
    
    columns = []
    for estimator in estimators:
        columns.append(estimator[0] + '_predict')
    
    test_predict = pd.DataFrame(np.zeros((m_test, n)), columns=columns)
    
    # 实例化重复交叉验证评估器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 执行交叉训练
    for estimator in estimators:
        model = estimator[1]
        oof_colName = estimator[0] + '_oof'
        predict_colName = estimator[0] + '_predict'
        
        for train_part_index, eval_index in kf.split(X, y):
            # 在训练集上训练
            X_train_part = X.loc[train_part_index]
            y_train_part = y.loc[train_part_index]
            model.fit(X_train_part, y_train_part)
            if blending == True:
                # 在留出集上进行预测并求均值
                train_oof[oof_colName] += model.predict_proba(X1)[:, 1] / n_splits
                # 在测试集上进行预测并求均值
                test_predict[predict_colName] += model.predict_proba(X_test)[:, 1] / n_splits
            else:
                # 在验证集上进行验证
                X_eval_part = X.loc[eval_index]
                # 将验证集上预测结果拼接入oof数据集
                train_oof[oof_colName].loc[eval_index] = model.predict_proba(X_eval_part)[:, 1]
                # 将测试集上预测结果填入predict数据集
                test_predict[predict_colName] += model.predict_proba(X_test)[:, 1] / n_splits
    
    # 添加标签列
    if blending == True:
        train_oof[y1.name] = y1
    else:
        train_oof[y.name] = y
        
    return train_oof, test_predict


def final_model_opt(final_model_l, param_space_l, X, y, test_predict):
    """
    Stacking元学习器自动优化与预测函数
    
    :param final_model_l: 备选元学习器组成的列表
    :param param_space_l: 备选元学习器各自超参数搜索空间组成的列表
    :param X: oof_train训练集特征
    :param y: oof_train训练集标签
    :param test_predict: 一级评估器输出的测试集预测结果
    
    :return：多组元学习器在oof_train上的最佳评分，以及最佳元学习器在test_predict上的预测结果
    """
    
    # 不同组元学习器结果存储列表
    # res_l用于存储模型在训练集上的评分
    res_l = np.zeros(len(final_model_l)).tolist()
    # test_predict_l用于存储模型在测试集test_predict上的预测结果
    test_predict_l = np.zeros(len(final_model_l)).tolist()
    
    for i, model in enumerate(final_model_l):
        # 输出元学习器单模预测结果
        # 执行网格搜索
        model_grid = GridSearchCV(estimator = model,
                                  param_grid = param_space_l[i],
                                  scoring='accuracy',
                                  n_jobs = 15)
        model_grid.fit(X, y)
        # 记录单模最佳模型，方便后续作为Bagging的基础评估器
        res1_best_model = model_grid.best_estimator_
        # 测试在训练oof数据集上的准确率
        res1 = model_grid.score(X, y)
        # 输出单模在test_predict上的预测结果
        res1_test_predict = model_grid.predict_proba(test_predict)[:, 1]
        
        # 输出元学习器交叉训练预测结果
        res2_temp = np.zeros(y.shape[0])
        res2_test_predict = np.zeros(test_predict.shape[0])
        # 交叉训练过程附带网格搜索以提升精度
        folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=12)
        for trn_idx, val_idx in folds.split(X, y):
            model_grid = GridSearchCV(estimator = model,
                                      param_grid = param_space_l[i],
                                      scoring='accuracy',
                                      n_jobs = 15)
            model_grid.fit(X.loc[trn_idx], y.loc[trn_idx])
            res2_temp += model_grid.predict_proba(X)[:, 1] / 10
            # 记录测试集上的预测结果
            res2_test_predict += model_grid.predict_proba(test_predict)[:, 1] / 10
        # 交叉训练模型组评分
        res2 = accuracy_score((res2_temp >= 0.5) * 1, y)

        # 元学习器的Bagging过程
        bagging_param_space = {"n_estimators": range(10, 21), 
                               "max_samples": np.arange(0.1, 1.1, 0.1).tolist()}
        
        bagging_final = BaggingClassifier(res1_best_model)
        BG = GridSearchCV(bagging_final, bagging_param_space, n_jobs=15).fit(X, y)
        # Bagging元学习器评分
        res3 = BG.score(X, y)
        # Bagging元学习器在测试集上评分
        res3_test_predict = BG.predict_proba(test_predict)[:, 1]
        
        # 三组模型评分组成列表
        res_l_temp = [res1, res2, res3]
        # 三组模型在测试集上预测结果组成列表
        test_predict_l_temp = [res1_test_predict, res2_test_predict, res3_test_predict]
        # 挑选评分最高模型
        best_res = np.max(res_l_temp)
        # 挑选评分最高模型输出的测试集概率预测结果
        best_test_predict = test_predict_l_temp[np.argmax(res_l_temp)]
        # 将最佳模型写入res_l对应位置
        res_l[i] = best_res
        # 将最佳模型在测试集上的评分写入test_predict_l
        test_predict_l[i] = best_test_predict
        
    # 再从res_l中选取训练集上最佳评分
    best_res_final = np.max(res_l) 
    # 根据训练集上的最佳评分，选取挑选最佳测试集预测结果
    best_test_predict_final = test_predict_l[np.argmax(res_l)]
    
    return best_res_final, best_test_predict_final

#######################################################
## Part 3.评估器默认参数空间

# 决策树一级评估器默认超参数空间
tree_params_space = {'tree_max_depth': hp.choice('tree_max_depth', np.arange(2, 20).tolist()), 
                     'tree_min_samples_split': hp.choice('tree_min_samples_split', np.arange(2, 15).tolist()), 
                     'tree_min_samples_leaf': hp.choice('tree_min_samples_leaf', np.arange(1, 15).tolist()), 
                     'tree_max_leaf_nodes': hp.choice('tree_max_leaf_nodes', np.arange(2, 51).tolist())}

# 随机森林一级评估器默认超参数空间
RF_params_space = {'RF_min_samples_leaf': hp.choice('RF_min_samples_leaf', np.arange(1, 20).tolist()), 
                   'RF_min_samples_split': hp.choice('RF_min_samples_split', np.arange(2, 20).tolist()), 
                   'RF_max_depth': hp.choice('RF_max_depth', np.arange(2, 20).tolist()), 
                   'RF_max_leaf_nodes': hp.choice('RF_max_leaf_nodes', np.arange(20, 200).tolist()), 
                   'RF_n_estimators': hp.choice('RF_n_estimators', np.arange(20, 200).tolist()), 
                   'RF_max_samples': hp.uniform('RF_max_samples', 0.2, 0.8)}

# 逻辑回归一级评估器默认超参数空间
lr_params_space = {'lr_C': hp.uniform('lr_C', 0, 1), 
                   'lr_penalty': hp.choice('lr_penalty', ['l1', 'l2']), 
                   'lr_thr': hp.uniform('lr_thr', 0, 1)}

# 逻辑回归元学习器默认超参数空间
lr_final_param = [{'thr': np.arange(0.1, 1.1, 0.1).tolist(), 
                   'penalty': ['l1'], 
                   'C': np.arange(0.1, 1.1, 0.1).tolist(), 
                   'solver': ['saga']}, 
                  {'thr': np.arange(0.1, 1.1, 0.1).tolist(), 
                   'penalty': ['l2'], 
                   'C': np.arange(0.1, 1.1, 0.1).tolist(), 
                   'solver': ['lbfgs', 'newton-cg', 'sag', 'saga']}]
# 决策树元学习器默认超参数空间
tree_final_param = {'max_depth': np.arange(2, 16, 1).tolist(), 
                    'min_samples_split': np.arange(1, 5, 1).tolist(), 
                    'min_samples_leaf': np.arange(1, 4, 1).tolist(), 
                    'max_leaf_nodes':np.arange(6, 30, 1).tolist()}

#######################################################
## Part 4.自动TPE超参数优化评估器

class tree_cascade(BaseEstimator, ClassifierMixin, TransformerMixin):
    
    def __init__(self, tree_params_space, max_evals=1000):
        self.tree_params_space = tree_params_space
        self.max_evals = max_evals
        
    def fit(self, X, y):
        def hyperopt_tree(params, train=True):
            # 读取参数
            if train == True:
                max_depth = params['tree_max_depth']
                min_samples_split = params['tree_min_samples_split']
                min_samples_leaf = params['tree_min_samples_leaf']
                max_leaf_nodes = params['tree_max_leaf_nodes']
            else: 
                max_depth = params['tree_max_depth'] + 2
                min_samples_split = params['tree_min_samples_split'] + 2
                min_samples_leaf = params['tree_min_samples_leaf'] + 1
                max_leaf_nodes = params['tree_max_leaf_nodes'] + 2

            # 实例化模型
            tree = DecisionTreeClassifier(max_depth=max_depth, 
                                          min_samples_split=min_samples_split, 
                                          min_samples_leaf=min_samples_leaf, 
                                          max_leaf_nodes=max_leaf_nodes, 
                                          random_state=12)

            if train == True:
                res = -cross_val_score(tree, X, y).mean()
            else:
                res = tree.fit(X, y)

            return res

        def param_hyperopt_tree(max_evals):
            params_best = fmin(fn = hyperopt_tree,
                               space = self.tree_params_space,
                               algo = tpe.suggest,
                               max_evals = max_evals)    

            return params_best
        
        tree_params_best = param_hyperopt_tree(self.max_evals)
        self.clf = hyperopt_tree(tree_params_best, train=False)
        return self
    
    def predict_proba(self, X):
        res_proba = self.clf.predict_proba(X)
        return res_proba
    
    def predict(self, X):
        res = self.clf.predict(X)
        return res
    
    def score(self, X, y):
        res = self.clf.score(X, y)
        return res
    
class RF_cascade(BaseEstimator, ClassifierMixin, TransformerMixin):
    
    def __init__(self, RF_params_space, max_evals=500):
        self.RF_params_space = RF_params_space
        self.max_evals = max_evals
        
    def fit(self, X, y):
        def hyperopt_RF(params, train=True):
            # 读取参数
            if train == True:
                min_samples_leaf = params['RF_min_samples_leaf']
                min_samples_split = params['RF_min_samples_split']
                max_depth = params['RF_max_depth']
                max_leaf_nodes = params['RF_max_leaf_nodes']
                n_estimators = params['RF_n_estimators']
                max_samples = params['RF_max_samples']
            else: 
                min_samples_leaf = params['RF_min_samples_leaf'] + 1
                min_samples_split = params['RF_min_samples_split'] + 2
                max_depth = params['RF_max_depth'] + 2
                max_leaf_nodes = params['RF_max_leaf_nodes'] + 20
                n_estimators = params['RF_n_estimators'] + 20
                max_samples = params['RF_max_samples']
            # 实例化模型
            RF = RandomForestClassifier(min_samples_leaf = min_samples_leaf, 
                                        min_samples_split = min_samples_split,
                                        max_depth = max_depth, 
                                        max_leaf_nodes = max_leaf_nodes, 
                                        n_estimators = n_estimators, 
                                        max_samples = max_samples, 
                                        random_state=9)
            if train == True:
                res = -cross_val_score(RF, X, y).mean()
            else:
                res = RF.fit(X, y)

            return res

        def param_hyperopt_RF(max_evals):
            params_best = fmin(fn = hyperopt_RF,
                               space = self.RF_params_space,
                               algo = tpe.suggest,
                               max_evals = max_evals)    

            return params_best
        
        RF_params_best = param_hyperopt_RF(self.max_evals)
        self.clf = hyperopt_RF(RF_params_best, train=False)
        return self
    
    def predict_proba(self, X):
        res_proba = self.clf.predict_proba(X)
        return res_proba
    
    def predict(self, X):
        res = self.clf.predict(X)
        return res
    
    def score(self, X, y):
        res = self.clf.score(X, y)
        
class lr_cascade(BaseEstimator, ClassifierMixin, TransformerMixin):
    
    def __init__(self, lr_params_space, max_evals=20):
        self.lr_params_space = lr_params_space
        self.max_evals = max_evals
        
    def fit(self, X, y):
        def hyperopt_lr(params, train=True):
            # 读取参数
            if train == True:
                C = params['lr_C']
                penalty = params['lr_penalty']
                thr = params['lr_thr']
            else: 
                C = params['lr_C']
                penalty = ['l1', 'l2'][params['lr_penalty']]
                thr = params['lr_thr']
            # 实例化模型
            lr = logit_threshold(C = C,  
                                 thr = thr, 
                                 penalty = penalty, 
                                 solver = 'saga', 
                                 max_iter = int(1e6))
            
            if train == True:
                res = -cross_val_score(lr, X, y).mean()
            else:
                res = lr.fit(X, y)

            return res

        def param_hyperopt_lr(max_evals):
            params_best = fmin(fn = hyperopt_lr,
                               space = self.lr_params_space,
                               algo = tpe.suggest,
                               max_evals = max_evals)    

            return params_best
        
        lr_params_best = param_hyperopt_lr(self.max_evals)
        self.clf = hyperopt_lr(lr_params_best, train=False)
        return self
    
    def predict_proba(self, X):
        res_proba = self.clf.predict_proba(X)
        return res_proba
    
    def predict(self, X):
        res = self.clf.predict(X)
        return res
    
    def score(self, X, y):
        res = self.clf.score(X, y)
        return res
    
