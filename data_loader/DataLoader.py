import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
import traceback


class DataLoader:
    """
    加载mysql数据
     Args:
       ip:mysql数据库ip
       username:mysql数据库账号
       password:mysql数据库密码
       port:mysql数据库端口号
       query_sql:查询sql

     Returns:
         dataframe

     """

def load_data_from_mysql(ip='localhost', username='root', password='1234', port=3306, query_sql=None):
    try:
        url = 'mysql+pymysql://{0}:{1}@{2}:{3}'.format(username, password, ip, port)
        print(url)
        engine = create_engine(url)
        df_data = pd.read_sql_query(query_sql, engine)
        engine.dispose()
        return df_data
    except Exception as e:
        print(str(e))
#         print(traceback.format_exc())


 """
    加载csv格式文件数据
     Args:
       file_path:文件实际路径
       index_col:索引列

     Returns:
         dataframe

     """
def load_data_from_csv(file_path='./', index_col=None):
    df_data = pd.read_csv(file_path, index_col=index_col)
    return df_data

 """
    加载excel格式文件数据
     Args:
       file_path:文件实际路径
       index_col:索引列

     Returns:
         dataframe

     """
def load_data_from_xls(file_path='./',index_col=None):
    df_data = pd.read_excel(file_path,index_col=index_col)
    return df_data