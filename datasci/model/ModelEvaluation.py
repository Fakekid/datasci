import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve, ShuffleSplit, validation_curve

warnings.filterwarnings("ignore")


def evaluate(estimator=None, X=None, y=None):
    """
      模型评估
    Args:
      estimator: 预估器实例
      X: 样本集
      y: 目标变量
    Returns:
       返回预估器评估指标字典

    Owner:wangyue29
    """
    pred_proba = estimator.predict_proba(X)[:, 1]
    pred = estimator.predict(X)
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


def roc(estimator=None, X=None, y=None):
    """
      roc 曲线
    Args:
      estimator: 预估器实例
      X: 样本集
      y: 目标变量
    Returns:
       返回roc曲线

    Owner:wangyue29
    """
    pred_proba = estimator.predict_proba(X)[:, 1]

    fpr, tpr, threshold = roc_curve(y, pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_learning_curve(estimator=None, X=None, y=None, ylim=(0.5, 1.01), cv=None,
                        n_splits=10, test_size=0.2, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
      学习曲线
    Args:
      estimator: 预估器实例
      X: 样本集
      y: 目标变量
      ylim: y轴坐标范围，默认从0.5到1.01
      cv: 交叉样本量，默认为空后，利用ShuffleSplit方法获取
      n_splits: 划分训练集、测试集的次数，默认为10，ShuffleSplit方法参数
      test_size: 测试集比例或样本数量，默认为0.2，ShuffleSplit方法参数
      n_jobs: CPU并行核数，默认为1，-1的时候，表示cpu里的所有core进行工作
      train_sizes: 训练样本比例，默认[0.1,0.325,0.55,0.775,1.]

    Returns:
       返回绘制学习曲线

    Owner:wangyue29
    """
    plt.figure()

    title = r"Learning Curve ({0})".format(estimator.__class__.__name__)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if cv is None:
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_validation_curve(estimator=None, X=None, y=None, param_name="gamma",
                          param_range=np.logspace(-6, -1, 5), scoring="accuracy",
                          n_jobs=1, ylim=(0.0, 1.01), verbose=0):
    """
      验证曲线
    Args:
      estimator: 预估器实例
      X: 样本集
      y: 目标变量
      param_name: 参数名称，默认gamma
      param_range: 训练样本比例，默认从0.0到1.01
      scoring: 评分函数，默认accuracy
      n_jobs: CPU并行核数，默认为1，-1的时候，表示cpu里的所有core进行工作
      ylim: y轴坐标范围，默认从0.5到1.01
      verbose: 是否打印调试信息，默认不打印

    Returns:
       返回绘制验证曲线

    Owner:wangyue29
    """

    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name,
                                                 param_range=param_range,
                                                 scoring=scoring, n_jobs=n_jobs, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    title = r"Validation Curve ({0})".format(estimator.__class__.__name__)
    plt.title(title)

    plt.xlabel(r"$\{0}$".format(param_name))
    plt.ylabel("Score")

    if ylim is not None:
        plt.ylim(*ylim)

    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
