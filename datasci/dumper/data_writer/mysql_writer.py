# -*- coding:utf8 -*-

from datasci.utils.mylog import get_stream_logger
from datasci.utils.mysql_utils import MysqlUtils

class MySQLDataWriter(object):
    """
        Mysql data writer
    """

    def __init__(self,
                 mysql_utils=MysqlUtils('Mysql-data_bank'),
                 sql='',
                 log=None
                 ):
        self.mysql_utils = mysql_utils
        self.sql = sql
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("DATA_WRITER", level=log_level) if log is None else log

    def save_data(self, result):
        try:
            self.mysql_utils.get_executemany_sql(self.sql, result)
        except Exception as e:
            self.log.error('SQL failed! Reason : %s ' % e)

    def __del__(self):
        del self.mysql_utils
