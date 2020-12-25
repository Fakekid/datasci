
import pandas as pd
from datasci.loader.data_reader.mysql_reader import MySQLDataReader


# Get data with stream
def get_mysql_stream_data(sql_file, batch_size=1000, max_iter=2, offset=0, batch_output=True):
    """
        Args
        -------
        sql_file
            SQL file path

        batch_size
            the batch size in every iterate

        max_iter
            max iterate

        offset
            the offset of data that start read

        batch_output
            output way

        Returns
        -------
        mysqlReader
            a iterater from mysql data source
    """

    with open(sql_file) as f:
        sql = f.read()
    mysqlReader = MySQLDataReader(
        sql=sql, batch_size=batch_size,
        max_iter=max_iter, offset=offset, batch_output=batch_output)
    return mysqlReader

def get_data(input_args):
    """
        Args
        -------
        input_args
            input config with different data source

        Returns
        -------
        pandas.Dataframe
    """
    data_type = input_args.get('data_type')
    data_args = input_args.get(data_type)
    if data_type == 'mysql':
        batch_size = data_args.get("args").get("batch_size")
        max_iter = data_args.get("args").get("max_iter")
        offset = data_args.get("args").get("offset")
        batch_output = data_args.get("args").get("batch_output")
        return get_mysql_stream_data(data_args.get('sql'), batch_size=batch_size, max_iter=max_iter, offset=offset,
                                     batch_output=batch_output)
    if data_type == 'file':
        path = data_args.get('path')
        return pd.read_csv(path)
