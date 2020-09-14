import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import warnings

warnings.filterwarnings("ignore")

# 默认模型列表
default_models = [('dt', {}), ('lr', {}), ('xgb', {})]

# 模型简称与模型实体映射关系
model_name_mapping = {
    'dt': DecisionTreeClassifier(random_state=42),
    'lr': LogisticRegression(),
    'rfc': RandomForestClassifier(random_state=42),
    'svc': SVC(kernel='rbf', gamma='scale'),
    'ada': AdaBoostClassifier(),
    'gbc': GradientBoostingClassifier(),
    "xgb": XGBClassifier()
}


def select_best_model(models=[], X=None, y=None, scoring='roc_auc', cv=5, verbose=0):
    """
       模型选择，基于scoring评分排名
     Args:
       models: 候选模型列表
       X: 样本集
       y: 目标变量
       scoring: 评分函数，默认roc_auc
       cv: 交叉验证，默认是5
       verbose: 是否打印信息，默认不打印

     Returns:
        返回最好的模型，模型评分结果集

     Owner:wangyue29
     """
    model_result = dict()
    best_model = None
    best_score = 0.0

    if 0 == len(models):
        models = default_models

    for model in models:
        model_name = model[0]
        model_params = model[1]
        clf = model_name_mapping[model_name]
        if 0 != len(model_params):
            print(model_params)
            clf.set_params(**model_params)

        clf.fit(X, y)

        model_full_name = clf.__class__.__name__
        scores = cross_val_score(clf, X, y, verbose=0, cv=cv, scoring=scoring)
        score_avg = scores.mean()
        model_result[model_full_name] = score_avg

        if 1 == verbose:
            print('Cross-validation of : {0}'.format(model_full_name))
            print('{0} {1:0.2f} (+/- {2:0.2f})'.format(scoring, score_avg, scores.std()))

        if score_avg >= best_score:
            best_model = clf
            best_score = score_avg

    print('Best Model: {0},Score:{1:0.2f}'.format(best_model.__class__.__name__, best_score))

    result = pd.DataFrame({
        'Model': [i for i in model_result.keys()],
        'Score': [i for i in model_result.values()]})

    result.sort_values(by='Score', ascending=False)
    return best_model, result


def grid_search_optimization(model, param_grid={}, X=None, y=None, scoring='roc_auc', cv=5, verbose=0):
    """
       模型优化-网格搜索
     Args:
       model: 模型实例
       param_grid: 模型参数
       X: 样本集
       y: 目标变量
       scoring: 评分函数，默认roc_auc
       cv: 交叉验证，默认是5
       verbose: 是否打印信息，默认不打印

     Returns:
        返回最好的模型，模型评分结果集

     Owner:wangyue29
     """

    cross_validation = StratifiedKFold(n_splits=cv)

    grid_search = GridSearchCV(
        model,
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


def feature_importances(model=None, X=None, thresholds=0.01, palette=None):
    """
      模型输出特征重要性
    Args:
      model: 模型实例
      X: 样本集
      thresholds: 特征重要性阈值，默认是0.01
      palette: 使用不同的调色板，默认是None

    Returns:
       返回特征重要性，按照降序排序

    Owner:wangyue29
    """
    importance_feature = list(zip(X.columns, model.feature_importances_))
    importance_feature = pd.DataFrame(importance_feature, columns=['feature_name', 'importances'])

    importance_feature = importance_feature.loc[importance_feature['importances'] >= thresholds]
    importance_feature = importance_feature.sort_values(by='importances', ascending=False)
    sns.set_style("darkgrid")
    sns.barplot(y='feature_name', x='importances', data=importance_feature, orient='h', palette=palette)
    plt.show()

    return importance_feature


def train(model_name='xgb', model_params={}, X=None, y=None):
    """
       构建训练模型
     Args:
       model_name: 模型名称，默认是xgb
       model_params: 模型参数
       X: 样本集
       y: 目标变量

     Returns:
        返回训练后的模型实例

     Owner:wangyue29
     """
    clf = model_name_mapping[model_name]

    if 0 != len(model_params):
        clf.set_params(**model_params)

    clf.fit(X, y)
    return clf


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
