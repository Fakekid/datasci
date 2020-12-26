# -*- coding:utf8 -*-
from datasci.utils.mylog import get_stream_logger
from pyspark.sql import SparkSession


def get_host_and_ip():
    """
       get host and ip
    """
    try:
        import socket

        hostname = socket.gethostname()  # 获取当前主机名
        # 通过hostname查询,注意这个并不一定会得到真确的IP地址
        ip = socket.gethostbyname(socket.gethostname())
        return hostname, ip
    except Exception as e:
        raise e
        # return "unknown_host", "0.0.0.0"


hostname, ip = get_host_and_ip()


class HiveSQLDataReader(object):
    """
        Hive data Reader
    """

    def __init__(self,
                 batch_size=1000000,
                 offset=0,
                 max_iter=10,
                 sql='',
                 func=None,
                 log=None
                 ):
        self.sql = sql
        self.batch_size = batch_size
        self.offset = offset
        self.max_iter = max_iter
        self.func = func
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("DATA_READER", level=log_level) if log is None else log

        builder = SparkSession.builder
        # 队列要切换成事业部自己的队列
        builder.config("spark.yarn.queue", 'root.wangxiao.xes1v1_dw')
        builder.config("spark.driver.host", ip)
        # 根据任务调整自己的资源到合理范围
        builder.config("spark.executor.memory", '5g')
        builder.config("spark.driver.memory", '5g')
        builder.config("spark.driver.maxResultSize", "5g")
        builder.config("spark.executor.instances", 2)
        builder.config("spark.executor.cores", 2)
        builder.config("spark.sql.adaptive.enabled", 'true')
        builder.config("hive.exec.dynamic.partition.mode", "nonstrict");
        # builder.master(master_url)
        self.spark = builder.enableHiveSupport().appName("action_sequence").getOrCreate()

    def read_data(self):
        """
            Read data
        """
        sql_xes_net_data = """
        select * from xes_1v1_data_bank.xes_net_1v1_user_action_sequence_sample where dt =20200830 limit 10
        """
        result_sql_xes_net_data = self.spark.sql(sql_xes_net_data)
        df_xes_net_data = result_sql_xes_net_data.toPandas()
        return df_xes_net_data
