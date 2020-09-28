# encoding:utf8

from setuptools import setup, find_packages

inner_packages = find_packages(include=['datasci.loader.*', 'datasci.eda.*',
                                        'datasci.model.*', 'datasci.preprocesing.*'])
setup(name='datasci',
      version='0.0.5',
      description='TAL WangXiao Data Science Toolkit',
      url='',
      author='wangyue,lianxiaolei',
      author_email='wangyue29@tal.com,lianxiaolei@tal.com',
      license='MIT',
      include_package_data=False,
      packages=['datasci.loader',
                'datasci.eda',
                'datasci.model',
                'datasci.preprocessing',
                'datasci.constant'] + inner_packages,
      install_requires=[  # 依赖列表
        'pandas>=1.0.5',
        'pymysql>=0.10.0',
        'numpy>=1.14.3',
        'seaborn==0.10.1',
        'numpy==1.19.1',
        'tqdm==4.47.0',
        'scikit_optimize>=0.8.1',
        'shap>=0.36.0',
        'xgboost>=1.2.0',
        'scipy>=1.5.2',
        'scikit_learn==0.22.1'
      ],
      zip_safe=False)
