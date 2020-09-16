import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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


def feature_importances(estimator=None, X=None, thresholds=0.01, palette=None):
    """
      预估器输出特征重要性
    Args:
      estimator: 预估器实例
      X: 样本集
      thresholds: 特征重要性阈值，默认是0.01
      palette: 使用不同的调色板，默认是None

    Returns:
       返回特征重要性，按照降序排序

    Owner:wangyue29
    """
    importance_feature = list(zip(X.columns, estimator.feature_importances_))
    importance_feature = pd.DataFrame(importance_feature, columns=['feature_name', 'importances'])

    importance_feature = importance_feature.loc[importance_feature['importances'] >= thresholds]
    importance_feature = importance_feature.sort_values(by='importances', ascending=False)
    sns.set_style("darkgrid")
    sns.barplot(y='feature_name', x='importances', data=importance_feature, orient='h', palette=palette)
    plt.show()

    return importance_feature


def train(estimator_name='XGB', estimator_params={}, X=None, y=None):
    """
       构建训练预估器
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


def plot_train_test_dataset_feature_dist(X_train=None, X_test=None, feature_names=[], f_rows=1, f_cols=2):
    """
       训练集、测试集的特征分布可视化
     Args:
       X_train: 训练集
       X_test: 测试集
       feature_names: 特征名称,默认可自动识别连续特征
       f_rows: 图行数，默认值1
       f_cols: 图列数，默认值2
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """

    if 0 == len(feature_names):
        feature_names = X_test.columns

    if 1 == f_rows and 0 != len(feature_names):
        f_rows = len(feature_names)

    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)

        ax = sns.kdeplot(X_train[feat_name], color='Red', shade=True)
        ax = sns.kdeplot(X_test[feat_name], color='Green', shade=True)

        ax.set_xlabel(feat_name)
        ax.set_ylabel('Frequency')
        ax = ax.legend(['train', 'test'])

    plt.show()
