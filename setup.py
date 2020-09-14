# encoding:utf8

from setuptools import setup, find_packages

inner_packages = find_packages(include=['datasci.data_loader.*', 'datasci.eda.*',
                                        'datasci.model.*', 'datasci.preprocesing.*'])
setup(name='datasci',
      version='0.0.2',
      description='TAL WangXiao Data Science Toolkit',
      url='',
      author='wangyue,lianxiaolei',
      author_email='wangyue29@tal.com,lianxiaolei@tal.com',
      license='MIT',
      include_package_data=False,
      packages=['datasci.data_loader',
                'datasci.eda',
                'datasci.model',
                'datasci.preprocesing',
                'datasci.constant'] + inner_packages,
      install_requires=[  # 依赖列表
        'pandas>=1.0.5',
        'pymysql>=0.10.0',
        'numpy>=1.14.3'
      ],
      zip_safe=False)
