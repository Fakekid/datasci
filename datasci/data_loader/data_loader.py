# coding:utf8

import numpy as np
import pandas as pd
import pymysql
import math
import os
from datasci.constant import VALUE_ERROR_VALUE_NOT_NONE_AT_SAME_TIME
import multiprocessing


def load_text_data(file_name, **kwargs):
  """
    Load text file.
  Args:
    file_name: a str value, file name.
    error: a str value, method of error operating.
    sep: a str value, segment char.

  Returns:
    if sep is not null, return a list[list[]] value, else return a list[str] value.

  """
  encoding = kwargs.get('encoding', 'utf8')
  error = kwargs.get('error', 'strict')
  sep = kwargs.get('sep')
  with open(file_name, 'r', encoding=encoding, errors=error) as fin:
    content = fin.readlines()
    if sep:
      content = [item.replace('\n', '').split(sep) for item in content]
    else:
      content = [item.replace('\n', '') for item in content]
  return content


def load_csv_data(file_name, **kwargs):
  """
    Load CSV data with pandas lib.
    For detail of full parameters, please go to https://www.cnblogs.com/traditional/p/12514914.html.
  Args:
    file_name:      a str value, file name.
    names:          a list value, column's names for data frame.
    header:         a int value, header row num.
    usecols:        a list value, columns you wanna use.
    index_col:      a str value, the column will be indexed.
    dtype:          a str or dict value, data type for each column.
    engine:         a str value, reading engine. default `c`, [`c`,`python`]
  Returns:
    a DataFrame value
  """

  names = kwargs.get('names')
  header = kwargs.get('header', 0)
  usecols = kwargs.get('usecols')
  index_col = kwargs.get('index_col')
  dtype = kwargs.get('dtype', 'object')
  engine = kwargs.get('engine', 'c')
  excel = kwargs.get('excel', False)

  if not excel:
    sep = kwargs.get('set', ',')
    data = pd.read_csv(file_name, engine=engine, sep=sep, names=names, header=header, usecols=usecols,
                       index_col=index_col, dtype=dtype)
  else:
    sheet_name = kwargs.get('sheet_name', 0)
    data = pd.read_excel(file_name, sheet_name=sheet_name, engine=engine, names=names, header=header, usecols=usecols,
                         index_col=index_col, dtype=dtype)

  return data


def load_np_data(file_name, **kwargs):
  """
    Load numpy file.
  Args:
    file_name:    a str value, file name.
    seg:          a str value, segment char.
    dtype:        a numpy dtype value, data type.

  Returns:
    a numpy array
  """
  is_txt = kwargs.get('is_txt')
  sep = kwargs.get('sep')
  dtype = kwargs.get('dtype')

  if is_txt:
    func = np.loadtxt
  else:
    func = np.load

  data = func(file_name, delimiter=sep, dtype=dtype)
  return data


def load_mysql_data(sql, **kwargs):
  """
    Load data from mysql using sql, and return a DataFrame with fixed columns.

  Args:
    sql:          a str value, execute the sql to get data from mysql.
    host:         a str value, host name.
    user:         a str value, mysql user account.
    password:     a str value, mysql password for given user name.
    db:           a str value, mysql database.
    charset:      a str value, database charset.
    columns:      a str or a list value, the columns you wanna use.

  Returns:
    a DataFrame value
  """
  host = kwargs.get('host')
  port = kwargs.get('port')
  user = kwargs.get('user')
  password = kwargs.get('password')
  db = kwargs.get('db')
  charset = kwargs.get('charset')

  assert host is not None and user is not None and password is not None, \
    ValueError('The params {host, user, password} are not allowed be null.')

  con = pymysql.connect(host=host, port=port, user=user, password=password, database=db, charset=charset)
  data = pd.read_sql(sql=sql, con=con)
  columns = kwargs.get('columns')
  if columns is not None:
    data = data[columns]

  return data


def load_data_multi_processor(idx, manager_dict, **kwargs):
  pass


class DataLoader(object):
  """

  """

  def __init__(self, data_path=None, data_array=None, data_processor=None, data_type='text', **kwargs):
    assert data_path is not None or data_array is not None, ValueError(VALUE_ERROR_VALUE_NOT_NONE_AT_SAME_TIME)

    self.data_type = data_type

    if data_array is not None:  # 传入的是数组
      pass
    else:  # 传入的是文件目录
      self.data_path = data_path

      self.is_dir = os.path.isdir(data_path)
      if not self.is_dir:
        data_array = self._load_data(**kwargs)
      else:
        num_processor_for_files = kwargs.get('num_processor_for_files', 1)

        file_names = os.listdir(data_path)

        if num_processor_for_files < 2:  # 单进程
          data_array = []
          for file_name in file_names:
            data_array_batch = self._load_data(file_name)
            data_array.append(data_array_batch)
          data_array = np.concatenate(data_array, axis=0)
        else:
          # 定义进程池
          pool = multiprocessing.Pool(num_processor_for_files)
          # 定义数据共享管理器
          data_manager = multiprocessing.Manager()
          data_array_dict = data_manager.dict()

          # 文件分批
          # 计算每个进程处理的数据量
          batch_size = math.ceil(len(file_names) / num_processor_for_files)
          # 按照单进程数据量对数据集进行切分
          batch_idxs = [list(range(batch_size * idx, min(batch_size * (idx + 1), len(file_names)))) for idx in
                        range(num_processor_for_files)]

          # 向进程池分发任务
          for idx in range(len(batch_idxs)):
            _ = pool.apply_async(load_data_multi_processor,
                                 (idx, data_array_dict,))
          pool.close()
          pool.join()

    # TODO pytorch里的初始化方法中有loder参数，即一个自定义的数据加载函数，这里打算改成data_processor
    pass

  def _load_data(self, data_path, **kwargs):
    pass
