# -*- coding:utf8 -*-
from datasci.utils.mylog import get_stream_logger
from sqlalchemy import create_engine
import pandas as pd
import time


class MySQLDataReader(object):
    """
        Mysql data reader (a Iterator)
    """
    def __init__(self,
                 engine=create_engine(
                     'mysql+pymysql://result_bigdata:l7oekleyZyygvsnhpU3@rm-2ze8s56qvnoda6o811o.mysql.rds.aliyuncs.com:3306/data_bank',
                     encoding='utf8'),
                 batch_size=1000000,
                 offset=0,
                 max_iter=10,
                 sql='',
                 func=None,
                 batch_output = False,
                 log=None
                 ):
        self.engine = engine
        self.sql = sql
        self.batch_size = batch_size
        self.offset = offset
        self.max_iter = max_iter
        self.func = func
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("DATA_READER", level=log_level) if log is None else log
        self.batch_output = batch_output

    def __iter__(self):
        begin = self.offset
        lastIter = False
        self.log.info('Iterate start ... ...')
        while self.max_iter > 0 or self.max_iter <= -1:
            if lastIter:
                self.log.info('Iterate over ... ...')
                break
            # 获取MySQL数据
            sql = '%s limit %s offset %s ' % (self.sql, self.batch_size, begin)
            try:
                df = pd.read_sql(sql, self.engine)
            except Exception as e:
                self.log.warn('SQL connect failed, skip this batch size !')
                continue
            # 判断最后一次迭代，并处理数据后退出
            cur_batch_size = len(df)
            self.log.debug('Batch size : %s' % cur_batch_size)
            if cur_batch_size == 0:
                self.log.info('Iterate over ... ...')
                break
            retlist = df.values.tolist()
            if cur_batch_size < self.batch_size:
                lastIter = True
                self.log.info(' start iterate : %s --> %s , start time : %s' % (
                begin, begin + cur_batch_size, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            else:
                self.log.info(' start iterate : %s --> %s , start time : %s' % (
                begin, begin + self.batch_size, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                begin = begin + self.batch_size
            if self.max_iter != -1:
                self.max_iter = self.max_iter - 1
            if self.batch_output:
                yield df
            else:
                for data in retlist:
                    if self.func is None:
                        yield data
                    else:
                        yield self.func(data)
