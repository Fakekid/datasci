import warnings

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")


def evaluate(model=None, X=None, y=None):
    """
      模型评估
    Args:
      model: 模型实例
      X: 样本集
      y: 目标变量
    Returns:
       返回模型评估指标字典

    Owner:wangyue29
    """
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


def roc(model=None, X=None, y=None):
    """
      roc 曲线
    Args:
      model: 模型实例
      X: 样本集
      y: 目标变量
    Returns:
       返回roc曲线

    Owner:wangyue29
    """
    pred_proba = model.predict_proba(X)[:, 1]

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
