# coding:utf8

import warnings

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV  # pip install scikit-optimize
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 默认预估器列表
default_estimators = [('LR', {}), ('XGB', {})]

# 预估器简称与预估器实体映射关系
estimator_name_mapping = {
    'NB': MultinomialNB(alpha=0.01),
    'DT': DecisionTreeClassifier(random_state=42),
    'LR': LogisticRegression(penalty='l2'),
    'KNN': KNeighborsClassifier(),
    'RFC': RandomForestClassifier(random_state=42),
    'SVC': SVC(kernel='rbf', gamma='scale'),
    'ADA': AdaBoostClassifier(),
    'GBDT': GradientBoostingClassifier(),
    "XGB": XGBClassifier(),
    "LGB": LGBMClassifier()
}


def select_best_estimator(estimators=[], X=None, y=None, scoring='roc_auc', cv=5, verbose=0):
    """
       选择最佳预估器，基于scoring评分排名
     Args:
       estimators: 候选预估器列表
       X: 样本集
       y: 目标变量
       scoring: 评分函数，默认roc_auc
       cv: 交叉验证，默认是5
       verbose: 是否打印调试信息，默认不打印

     Returns:
        返回最好的预估器，预估器评分结果集

     Owner:wangyue29
     """
    estimator_result = dict()
    best_estimator = None
    best_score = 0.0

    if 0 == len(estimators):
        estimators = default_estimators

    for estimator in estimators:
        estimator_name = estimator[0]
        estimator_params = estimator[1]
        estimator = estimator_name_mapping[estimator_name]

        if estimator is None:
            print ('wrong estimator name!')

        if 0 != len(estimator_params):
            print(estimator_params)
            estimator.set_params(**estimator_params)

        estimator.fit(X, y)

        estimator_full_name = estimator.__class__.__name__
        scores = cross_val_score(estimator, X, y, verbose=0, cv=cv, scoring=scoring)
        score_avg = scores.mean()
        estimator_result[estimator_full_name] = score_avg

        if 1 == verbose:
            print('Cross-validation of : {0}'.format(estimator_full_name))
            print('{0} {1:0.2f} (+/- {2:0.2f})'.format(scoring, score_avg, scores.std()))

        if score_avg >= best_score:
            best_estimator = estimator
            best_score = score_avg

    print('Best Model: {0},Score:{1:0.2f}'.format(best_estimator.__class__.__name__, best_score))

    result = pd.DataFrame({
        'Model': [i for i in estimator_result.keys()],
        'Score': [i for i in estimator_result.values()]})

    result.sort_values(by='Score', ascending=False)
    return best_estimator, result


def grid_search_optimization(estimator, param_grid={}, X=None, y=None, scoring='roc_auc', cv=5, verbose=0):
    """
       预估器优化-网格搜索
     Args:
       estimator: 预估器实例
       param_grid: 预估器参数
       X: 样本集
       y: 目标变量
       scoring: 评分函数，默认roc_auc
       cv: 交叉验证，默认是5
       verbose: 是否打印调试信息，默认不打印

     Returns:
        返回最好的预估器，预估器评分结果集

     Owner:wangyue29
     """

    cross_validation = StratifiedKFold(n_splits=cv)

    grid_search = GridSearchCV(
        estimator,
        scoring=scoring,
        param_grid=param_grid,
        cv=cross_validation,
        n_jobs=-1,
        verbose=verbose)

    grid_search.fit(X, y)

    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    return parameters


def bayesian_search_optimization(estimator, param_grid={}, X=None, y=None, n_iter=30, verbose=0):
    """
       预估器优化-贝叶斯搜索
     Args:
       estimator: 预估器实例
       param_grid: 预估器参数
       X: 样本集
       y: 目标变量
       n_iter: 迭代次数，默认30次
       verbose: 是否打印调试信息，默认不打印

     Returns:
        返回最好的预估器，预估器评分结果集

     Owner:wangyue29
     """

    bayes = BayesSearchCV(estimator, param_grid, n_iter=n_iter, random_state=42, verbose=verbose)
    bayes.fit(X, y)

    # best parameter combination
    parameters = bayes.best_params_

    # all combinations of hyperparameters
    bayes.cv_results_['params']

    # average scores of cross-validation
    bayes.cv_results_['mean_test_score']

    print('Best score: {}'.format(bayes.best_score_))
    print('Best parameters: {}'.format(bayes.best_params_))

    return parameters


def train(estimator_name='XGB', estimator_params={}, X=None, y=None):
    """
       训练预估器【支持分类、回归、聚类等】
     Args:
       estimator_name: 预估器名称，默认是XGB
       estimator_params: 预估器参数
       X: 样本集
       y: 目标变量

     Returns:
        返回训练后的预估器实例

     Owner:wangyue29
     """

    estimator = estimator_name_mapping[estimator_name]

    if estimator is None:
        print ('wrong estimator name!')

    if 0 != len(estimator_params):
        estimator.set_params(**estimator_params)

    estimator.fit(X, y)

    return estimator


def save(estimmator=None, filename=None, compress=3, protocol=None):
    """
       模型保存
     Args:
       estimator_name: 预估器名称
       filename: 模型保存路径，文件格式支持‘.z’, ‘.gz’, ‘.bz2’, ‘.xz’ or ‘.lzma’
       compress: 压缩数据等级，0-9，值越大，压缩效果越好，但会降低读写效率，默认建议是3
       protocol: pickle protocol,

     Returns:
        返回存储数据文件列表
    """
    filenames = joblib.dump(estimmator, filename, compress, protocol)

    return filenames
