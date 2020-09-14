import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import warnings

warnings.filterwarnings("ignore")

# 默认模型列表
default_models = [('dt', {}), ('lr', {}), ('rfc', {'n_estimators': 80})]

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


def feature_importances(model=None, X=None, thresholds=0.03, palette=None):
    importance_feature = list(zip(X.columns, model.feature_importances_))
    importance_feature = pd.DataFrame(importance_feature, columns=['feature_name', 'importances'])

    importance_feature = importance_feature.loc[importance_feature['importances'] >= thresholds]
    importance_feature = importance_feature.sort_values(by='importances', ascending=False)
    sns.set_style("darkgrid")
    sns.barplot(y='feature_name', x='importances', data=importance_feature, orient='h', palette=palette)
    plt.show()

    return importance_feature


def evaluate(model=None, X=None, y=None):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    evaluate_index = {}

    evaluate_index['auc'] = '%.3f' % metrics.roc_auc_score(y, pred_proba)
    evaluate_index['f1'] = '%.3f' % metrics.f1_score(y, pred)
    evaluate_index['recall'] = '%.3f' % metrics.recall_score(y, pred)
    evaluate_index['precision'] = '%.3f' % metrics.precision_score(y, pred)
    evaluate_index['accuracy'] = '%.3f' % metrics.accuracy_score(y, pred)
    evaluate_index['ture_rate'] = '%.3f' % (pred.sum() / pred.shape[0])
    evaluate_index['positive'] = '%s' % (pred.sum())
    #     print(metrics.classification_report(y,pred))
    return evaluate_index


def train(model_name='xgb', model_params={}, X=None, y=None):
    clf = model_name_mapping[model_name]

    if 0 != len(model_params):
        clf.set_params(**model_params)

    clf.fit(X, y)
    return clf
