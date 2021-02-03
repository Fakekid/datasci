import os
import time
from datasci.dao.mysql_dao import MySQLDao
from datasci.dao.bean.mysql_conf import MySQLConf
from datasci.loader.data_reader.batch_reader import MySQLDataReader
from datasci.utils.mysql_utils import MysqlUtils
from datasci.workflow.node.base import BaseNode
from datasci.workflow.output.join import JoinProcesser
from datasci.workflow.output.save import SaveProcesser


class MysqlReadNode(BaseNode):

    def run(self):
        sql = self.run_params.get('sql', None) if self.run_params is not None else None
        section = self.run_params.get('section', None) if self.run_params is not None else None
        if os.path.exists(sql):
            with open(sql) as f:
                sql = f.read()
        reader = MySQLDataReader(section=section, sql=sql)
        self.output_data = reader.read_data()
        self.is_finished = True


class MysqlExecNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        section = self.run_params.get('section', None) \
            if self.run_params is not None else "Mysql-data_bank"
        sql = self.run_params.get('sql', None) \
            if self.run_params is not None else None

        if os.path.exists(sql):
            with open(sql) as f:
                sql = f.read(sql)

        mysql_utils = MysqlUtils(section)
        result = [tuple(x) for x in self.input_data.values]
        try:
            if sql is not None:
                mysql_utils.get_executemany_sql(sql, result)
        except Exception as e:
            print(e)
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class MysqlUpdateNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        section = self.run_params.get('section', None) if self.run_params is not None else None
        from datasci.utils.read_config import get_global_config
        host = get_global_config(section, 'host')
        port = int(get_global_config(section, 'port'))
        user = get_global_config(section, 'user')
        password = get_global_config(section, 'password')
        db = get_global_config(section, 'db')
        charset = get_global_config(section, 'charset')
        mysql_conf = MySQLConf(host=host, port=port, user=user, passwd=password, db_name=db, charset=charset)
        mysql_dao = MySQLDao(mysql_conf)
        condition_cols = self.run_params.get('condition_cols',
                                             None) if self.run_params is not None else None
        condition_values = self.run_params.get('condition_values',
                                               None) if self.run_params is not None else self.input_data.loc[:,
                                                                                         condition_cols].values.tolist()
        table_name = self.run_params.get('table_name', None) if self.run_params is not None else None
        target_cols = self.run_params.get('target_cols',
                                          None) if self.run_params is not None else None
        self.input_data = self.input_merge()
        update_data = self.input_data.loc[:, target_cols].values.tolist()
        mysql_dao.update_data(table_name=table_name, condition_cols=condition_cols,
                              condition_values=condition_values,
                              target_cols=target_cols, target_values=update_data)
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class MysqlInsertNode(BaseNode):

    def run(self):
        section = self.run_params.get('section', None) if self.run_params is not None else None
        from datasci.utils.read_config import get_global_config
        host = get_global_config(section, 'host')
        port = int(get_global_config(section, 'port'))
        user = get_global_config(section, 'user')
        password = get_global_config(section, 'password')
        db = get_global_config(section, 'db')
        charset = get_global_config(section, 'charset')
        mysql_conf = MySQLConf(host=host, port=port, user=user, passwd=password, db_name=db, charset=charset)
        mysql_dao = MySQLDao(mysql_conf)
        target_cols = self.run_params.get('target_cols',
                                          None) if self.run_params is not None else self.input_data.columns.tolist()
        table_name = self.run_params.get('table_name', None) if self.run_params is not None else None
        update_col_when_duplicate = self.run_params.get('update_col_when_duplicate',
                                                        None) if self.run_params is not None else None
        duplicate_col_op = self.run_params.get('duplicate_col_op',
                                               None) if self.run_params is not None else None
        self.input_data = self.input_merge()
        data = self.input_data.loc[:, target_cols].values.tolist()
        mysql_dao.insert_data(table_name=table_name, cols=target_cols, values=data,
                              update_col_when_duplicate=update_col_when_duplicate, duplicate_col_op=duplicate_col_op)
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class DataJoinNode(BaseNode):

    def run(self):
        join_key = self.run_params.get('join_key', None) if self.run_params is not None else None
        if self.input_data is not None:
            if isinstance(self.input_data[0], dict):
                self.input_data = self.input_merge()
            join_class = JoinProcesser(
                **self.node_class_params) if self.node_class_params is not None else JoinProcesser()
            result = join_class.run(data=self.input_data, join_key=join_key)
        else:
            result = None
        self.output_data = result
        self.is_finished = True
        return self.output_data


class SelectDataFromDict(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        if not isinstance(self.input_data, dict):
            self.output_data = self.input_data
        else:
            keys = list(self.input_data.keys())
            tag = self.run_params.get('tag', None) \
                if self.run_params is not None else keys[0]
            self.output_data = self.input_data.get(tag, None)
        self.is_finished = True
        return self.output_data


class SelectColumnsNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        columns = self.run_params.get('columns', None) \
            if self.run_params is not None else self.input_data.columns.tolist()
        if columns is not None:
            self.output_data = self.input_data[columns]
        self.is_finished = True
        return self.output_data


class SaveNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        save_class = SaveProcesser(
            **self.node_class_params) if self.node_class_params is not None else SaveProcesser()

        output_config = self.run_params.get('output_config', None) if self.run_params is not None else None
        pagesize = self.run_params.get('pagesize', 10000) if self.run_params is not None else 10000
        model_name = self.run_params.get('model_name', 'default') if self.run_params is not None else 'default'

        ex_col = {
            'model_version': model_name,
            'dt': "%s" % time.strftime("%Y%m%d", time.localtime())
        }
        save_class.run(data=self.input_data, extend_columns=ex_col, output_config=output_config, pagesize=pagesize)
        self.is_finished = True
        return self.output_data
