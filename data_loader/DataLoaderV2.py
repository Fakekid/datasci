# coding:utf8

import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
import traceback
import math
import os
from constant import VALUE_ERROR_VALUE_NOT_NONE_AT_SAME_TIME
import multiprocessing


def load_text_data(self, file_name, **kwargs):
  pass


def load_pandas_data(self, file_name, **kwargs):
  pass


def load_np_data(self, file_name, **kwargs):
  pass


def load_mysql_data(self, sql, mysql_conf):
  pass


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
