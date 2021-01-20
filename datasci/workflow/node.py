import os
import time

from datasci.utils.mysql_utils import MysqlUtils

from datasci.workflow.output.save import JoinProcesser, SaveProcesser
from datasci.workflow.predict.predict import PredictProcesser
from datasci.workflow.train.train import TrainProcesser


class TrainNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
        train_class = TrainProcesser(
            **self.node_class_params) if self.node_class_params is not None else TrainProcesser()
        multi_process = self.run_params.get('multi_process', False) if self.run_params is not None else False
        train_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = None
        self.is_finished = True
        return self.output_data


class PredictNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
        predict_class = PredictProcesser(
            **self.node_class_params) if self.node_class_params is not None else PredictProcesser()
        multi_process = self.run_params.get('multi_process', False) if self.run_params is not None else False
        result = predict_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = result
        self.is_finished = True
        return self.output_data


class JoinNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
        join_class = JoinProcesser(
            **self.node_class_params) if self.node_class_params is not None else JoinProcesser()

        join_key = self.run_params.get('join_key', None) if self.run_params is not None else None
        join_label = self.run_params.get('join_label', 'join') if self.run_params is not None else 'join'
        result = join_class.run(data_dict=self.input_data, join_key=join_key)
        result[join_label] = join_class.join(result)
        self.output_data = result
        self.is_finished = True
        return self.output_data


class SaveNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
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


class SelectDataFromDict(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
        if not isinstance(self.input_data, dict):
            self.output_data = self.input_data
        else:
            keys = list(self.input_data.keys())
            tag = self.node_class_params.get('tag', None) \
                if self.node_class_params is not None else keys[0]
            self.output_data = self.input_data.get(tag, None)
        self.is_finished = True
        return self.output_data


class SelectColumnsNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
        columns = self.node_class_params.get('columns', None) \
            if self.node_class_params is not None else self.input_data.columns.tolist()
        if columns is not None:
            self.output_data = self.input_data[columns]
        self.is_finished = True
        return self.output_data


class MysqlExecNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
        section = self.node_class_params.get('section', None) \
            if self.node_class_params is not None else "Mysql-data_bank"
        sql = self.node_class_params.get('sql', None) \
            if self.node_class_params is not None else None

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


class StartNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
        print("Job start at %s " % time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class EndNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False

    def run(self):
        self.output_data = self.input_data
        self.is_finished = True
        print("Job finished at %s " % time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        return self.output_data
