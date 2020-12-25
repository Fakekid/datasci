# -*- coding:utf8 -*-

from datasci.dumper.data_writer.datawriter import DataWriter
from datasci.utils.mylog import get_stream_logger
from datasci.utils.mysql_utils import MysqlUtils


class MySQLDataWriter(DataWriter):
    """
        Mysql data writer
    """

    def __init__(self,
                 mysql_utils=MysqlUtils('Mysql-data_bank'),
                 sql='',
                 log=get_stream_logger('MySQLDataWriter')
                 ):
        self.mysql_utils = mysql_utils
        self.sql = sql
        self.log = log if log else get_stream_logger('MySQLDataReader')

    def to_mysql(self, result):
        try:
            self.mysql_utils.get_executemany_sql(self.sql, result)
        except Exception as e:
            self.log.warn('SQL connect failed! Skipping this record')


    def __del__(self):
        del self.mysql_utils
