
from sqlalchemy import MetaData, Table, create_engine

engine_data_bank = create_engine(
    'mysql+pymysql://result_bigdata:l7oekleyZyygvsnhpU3@rm-2ze8s56qvnoda6o811o.mysql.rds.aliyuncs.com:3306/data_bank',
    encoding='utf8')
engine_data_doris = create_engine(
    'mysql+pymysql://xes1v1:1v1.2v2@101.200.80.169:9030/xes1v1_db',
    encoding='utf8')


def save_data(data, output, args=None):
    """
    Save data
    """
    data_type = output.get('data_type')
    if data_type == 'file':
        path = output.get(data_type).get('path')
        data.to_csv('%s_%s' % (path, args))
    if data_type == 'mysql':
        sql = output.get(data_type).get('table')
        data.to_sql(sql, engine_data_bank, if_exists='append', index=False)
    if data_type == 'doris':
        sql = output.get(data_type).get('table')
        data.to_sql(sql, engine_data_doris, if_exists='append', index=False)
