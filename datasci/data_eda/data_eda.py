import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def data_distirbution_info(dataset):
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
        null_value_ratio = null_value / dataset[feat_name].shape[0]
        print ('特征名称:{},空值数:{},空值率:{:0.2f}'.format(feat_name, null_value, null_value_ratio))


def label_distribution(dataset, label='label'):
    """
       正、负样本分布信息统计
     Args:
       dataset: dataframe数据集
       label: 目标变量label值

     Returns:
        返回正、负样本分布信息结果
     Owner:wangyue29
     """
    pos_size = dataset[label].sum()
    neg_size = dataset[label].shape[0] - pos_size
    return '正样本数:{},负样本数:{},负样本数/正样本数:{:0.2f}'.format(pos_size, neg_size, pos_size / neg_size)


def categorical_feature_count_plot(dataset, feature_names=[], rotation=0):
    """
       离散特征条形图可视化
     Args:
       dataset: dataframe数据集
       feature_names: a list value,默认可自动识别离散特征
       rotation:横坐标值旋转角度，默认是0
     Returns:
        可视化呈现结果
     Owner:wangyue29
     """

    def count_plot(x, **kwargs):
        sns.countplot(x=x)
        plt.xticks(rotation=rotation)

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['object', 'category', 'bool', 'string']).columns

    f = pd.melt(dataset, value_vars=feature_names)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
    g.map(count_plot, "value")


def numberical_feature_dist_plot(dataset, feature_names=[], f_rows=2, f_cols=2, kde=True):
    """
       连续特征直方图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称,默认可自动识别连续特征
       f_rows: 图行数，默认值2
       f_cols: 图列数，默认值2
       kde:KDE分布，默认值True
       rotation:横坐标值旋转角度，默认是0
     Returns:
        可视化呈现结果
     Owner:wangyue29
     """
    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    f_rows = len(feature_names)
    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.distplot(dataset[feat_name], fit=stats.norm, kde=kde)

        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        res = stats.probplot(dataset['sepal_length'], plot=plt)

    plt.tight_layout()
    plt.show()


def numberical_feature_dist_plot_without_qq(dataset, kde=False, feature_names=[], rotation=0):
    """
       连续特征直方图可视化
     Args:
       dataset: dataframe数据集
       feature_names: a list value,默认可自动识别连续特征
       rotation:横坐标值旋转角度，默认是0
     Returns:
        可视化呈现结果
     Owner:wangyue29
     """

    def dist_plot(x, **kwargs):
        sns.distplot(x, kde=kde);
        plt.xticks(rotation=0)

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    f = pd.melt(dataset, value_vars=feature_names)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
    g.map(dist_plot, "value")


def numberical_feature_corr_heatmap_plot(dataset, feature_names=[]):
    """
       连续特征相关热力图可视化
     Args:
       dataset: dataframe数据集
       feature_names: a list value,默认可自动识别连续特征.
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


def linear_reg_corr_plot(dataset, feature_names=[], label='label', f_rows=0, f_cols=2):
    """
       线性回归关系图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称,默认可自动识别连续特征
       f_rows: 图行数，默认值2
       f_cols: 图列数，默认值2
     Returns:
        可视化呈现结果
     Owner:wangyue29
     """

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    f_rows = len(feature_names)
    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.regplot(x=feat_name, y=label, data=dataset, ax=ax,
                    scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                    line_kws={'color': 'k'});

        plt.xlabel(feat_name)
        plt.ylabel(label)

        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.distplot(dataset[feat_name].dropna())
        plt.xlabel(feat_name)
