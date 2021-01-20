# encoding:utf8

from setuptools import setup, find_packages

inner_packages = find_packages(include=['datasci.loader.*', 'datasci.eda.*','datasci.dumper.*','datasci.workflow.*',
                                        'datasci.model.*', 'datasci.preprocesing.*','datasci.utils.*',
                                        'datasci.dao.*'])
setup(name='datasci',
      namespace_packages=['datasci'],
      version='0.1.6',
      description='TAL WangXiao 1V1 Data Science Toolkit',
      url='',
      author='wangyue,lianxiaolei,zhaolihan,sunpeng,baijiaqi,lvhaiyan',
      author_email='wangyue29@tal.com,lianxiaolei@tal.com,zhaolihan@tal.com',
      license='MIT',
      include_package_data=False,
      packages=['datasci.loader',
                'datasci.loader.data_reader',
                'datasci.dumper',
                'datasci.dumper.data_writer',
                'datasci.eda',
                'datasci.model',
                'datasci.preprocessing',
                'datasci.constant',
                'datasci.dao',
                'datasci.workflow',
                'datasci.utils',
                'datasci.dao.bean'] + inner_packages,
      install_requires=[  # 依赖列表
          'pandas>=1.0.5',
          'pymysql>=0.10.0',
          'numpy>=1.14.3',
          'seaborn==0.10.1',
          'tqdm==4.47.0',
          'scikit_optimize>=0.8.1',
          'shap>=0.36.0',
          'xgboost>=0.71',
          'scipy>=1.4.1',
          'scikit_learn==0.22.1',
          'sqlalchemy>=1.3.22'
      ],
      zip_safe=False)
