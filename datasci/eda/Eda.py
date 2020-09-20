# coding:utf8

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

warnings.filterwarnings("ignore")


def data_dist_info(dataset):
    """
       查看数据分布
     Args:
       dataset: dataframe数据集

     Returns:

     Owner:wangyue29
     """

    print('--- 离散特征分布--- ')
    categorical_feature_names = dataset.select_dtypes(include=['object', 'category', 'bool', 'string']).columns
    for cat_fea in categorical_feature_names:
        print(cat_fea + '的特征分布如下')
        print('{}特征有个{}不同值'.format(cat_fea, dataset[cat_fea].nunique()))
        print(dataset[cat_fea].value_counts() + '\n')

    print('\n')

    print('--- 连续特征分布--- ')
    numberical_feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns
    for numberical_fea in numberical_feature_names:
        print(numberical_fea + '的特征分布如下')
        print('{:15}'.format(numberical_fea),
              'Skewness:{:05.2f}'.format(dataset[numberical_fea].skew()),
              '',
              'Kurtosis:{:06.2f}'.format(dataset[numberical_fea].kurt())
              )

    print('\n')
    print('--- dataset describe ---')
    print (dataset.describe())
    print('\n')
    print('--- dataset info ---')
    print (dataset.info(), '\n')


def null_value_info(dataset, feature_names=[]):
    """
       空值数、率信息统计
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表，默认所有特征名称

     Returns:
        返回空值统计信息结果

     Owner:wangyue29
     """

    if 0 == len(feature_names):
        feature_names = dataset.columns

    for feat_name in feature_names:
        null_value = dataset[feat_name].isnull().sum()
        null_value_ratio = null_value * 1.0 / dataset[feat_name].shape[0]
        print ('特征名称:{},空值数:{},空值率:{:0.2f}'.format(feat_name, null_value, null_value_ratio))


def target_dist_info(dataset, target='label'):
    """
       正、负样本分布信息统计
     Args:
       dataset: dataframe数据集
       target: 目标变量target值,默认'label'

     Returns:
        返回正、负样本分布信息结果

     Owner:wangyue29
     """
    pos_size = dataset[target].sum()
    neg_size = dataset[target].shape[0] - pos_size
    return '正样本数:{},负样本数:{},负样本数/正样本数:{:0.2f}'.format(pos_size, neg_size, pos_size / neg_size)


def plot_categorical_feature_bar_chart(dataset, feature_names=[], hue=None, f_rows=1, f_cols=2, palette=None):
    """
       离散特征条形图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表，默认可自动识别离散特征
       hue: 在x或y标签划分的同时，再以hue标签划分统计个数
       f_rows: 图行数，默认值1
       f_cols: 图列数，默认值2
       palette: 使用不同的调色板，默认是None
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['object', 'category', 'bool', 'string']).columns

    if 1 == f_rows and 0 != len(feature_names):
        f_rows = len(feature_names)

    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.countplot(x=feat_name, hue=hue, data=dataset, palette=palette)
        plt.title('variable={}'.format(feat_name))
        plt.xlabel('')

    plt.tight_layout()
    plt.show()


def plot_numberical_feature_hist(dataset, feature_names=[], f_rows=1, f_cols=2, kde=True, rotation=30):
    """
       连续特征直方图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表,默认可自动识别连续特征
       f_rows: 图行数，默认值1
       f_cols: 图列数，默认值2
       kde:KDE分布，默认值True
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """
    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    if 1 == f_rows and 0 != len(feature_names):
        f_rows = len(feature_names)

    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.distplot(dataset[feat_name], fit=stats.norm, kde=kde)
        plt.title('variable={}'.format(feat_name))
        plt.xlabel('')

        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        res = stats.probplot(dataset[feat_name], plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(dataset[feat_name])))

    plt.tight_layout()
    plt.xticks(rotation=rotation)
    plt.show()


def plot_numberical_feature_hist_without_qq_chart(dataset, kde=False, feature_names=[], rotation=0):
    """
       连续特征直方图可视化【无Q-Q图】
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表，默认可自动识别连续特征
       rotation:横坐标值旋转角度，默认是0
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """

    def dist_plot(x, **kwargs):
        sns.distplot(x, kde=kde)
        plt.xticks(rotation=0)

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    f = pd.melt(dataset, value_vars=feature_names)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
    g.map(dist_plot, "value")


def plot_numberical_feature_corr_heatmap(dataset, feature_names=[]):
    """
       连续特征相关热力图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表，默认可自动识别连续特征
     Returns:
        可视化呈现结果

     Owner:wangyue29
    """
    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    corr = dataset[feature_names].corr()

    f, ax = plt.subplots(figsize=(7, 7))
    plt.title('Correlation of Numberical Features', y=1, size=16)
    sns.heatmap(corr, annot=True, square=True, vmax=1.0, vmin=-1.0,
                linewidths=.5, annot_kws={'size': 12, 'weight': 'bold', 'color': 'blue'})


def plot_linear_reg_corr(dataset, feature_names=[], target='label', f_rows=1, f_cols=2, is_display_distplot=False):
    """
       线性回归关系图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称,默认可自动识别连续特征
       target: 目标变量target值,默认'label'
       f_rows: 图行数，默认值1
       f_cols: 图列数，默认值2
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    if 1 == f_rows and 0 != len(feature_names):
        f_rows = len(feature_names)

    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.regplot(x=feat_name, y=target, data=dataset, ax=ax,
                    scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                    line_kws={'color': 'k'})

        plt.title('variable=' + '{}'.format(feat_name))
        plt.xlabel('')
        plt.ylabel(target)

        if is_display_distplot:
            idx += 1
            ax = plt.subplot(f_rows, f_cols, idx)
            sns.distplot(dataset[feat_name].dropna())
            plt.xlabel(feat_name)
