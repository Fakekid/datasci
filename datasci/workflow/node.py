import os
import time
from datasci.utils.mysql_utils import MysqlUtils
from datasci.workflow.base_node import BaseNode
from datasci.workflow.output.join import JoinProcesser
from datasci.workflow.output.save import SaveProcesser
from datasci.workflow.predict.predict import PredictProcesser
from datasci.workflow.train.train import TrainProcesser


class TrainNode(BaseNode):

    def run(self):
        if self.input_data is not None:
            self.input_data = self.input_merge(axis=0)
        train_class = TrainProcesser(
            **self.node_class_params) if self.node_class_params is not None else TrainProcesser()
        multi_process = self.run_params.get('multi_process', False) if self.run_params is not None else False
        train_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = None
        self.is_finished = True
        return self.output_data


class PredictNode(BaseNode):

    def run(self):
        if self.input_data is not None:
            self.input_data = self.input_merge(axis=0)
        predict_class = PredictProcesser(
            **self.node_class_params) if self.node_class_params is not None else PredictProcesser()
        multi_process = self.run_params.get('multi_process', False) if self.run_params is not None else False
        result = predict_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = result
        self.is_finished = True
        return self.output_data


class JoinNode(BaseNode):

    def run(self):
        join_key = self.run_params.get('join_key', None) if self.run_params is not None else None
        if isinstance(self.input_data[0], dict):
            self.input_data = self.input_merge()
        join_class = JoinProcesser(
            **self.node_class_params) if self.node_class_params is not None else JoinProcesser()
        result = join_class.run(data=self.input_data, join_key=join_key)

        self.output_data = result
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


class SelectDataFromDict(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        if not isinstance(self.input_data, dict):
            self.output_data = self.input_data
        else:
            keys = list(self.input_data.keys())
            tag = self.node_class_params.get('tag', None) \
                if self.node_class_params is not None else keys[0]
            self.output_data = self.input_data.get(tag, None)
        self.is_finished = True
        return self.output_data


class SelectColumnsNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        columns = self.node_class_params.get('columns', None) \
            if self.node_class_params is not None else self.input_data.columns.tolist()
        if columns is not None:
            self.output_data = self.input_data[columns]
        self.is_finished = True
        return self.output_data


class MysqlExecNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
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


class StartNode(BaseNode):

    def run(self):
        from datasci.utils.mylog import get_stream_logger
        from datasci.workflow.config.log_config import log_level
        log = get_stream_logger("START NODE", level=log_level)
        log.info("Job start at %s " % time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class EndNode(BaseNode):

    def run(self):
        is_merge = self.run_params.get('is_merge', None) if self.run_params is not None else False
        axis = self.run_params.get('axis', None) if self.run_params is not None else 0
        if is_merge:
            self.output_data = self.input_merge(axis=axis)
        else:
            self.output_data = self.input_data
        self.is_finished = True
        from datasci.utils.mylog import get_stream_logger
        from datasci.workflow.config.log_config import log_level
        log = get_stream_logger("END NODE", level=log_level)
        log.info("Job finished at %s " % time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        return self.output_data


class DebugNode(BaseNode):

    def run(self):
        # node_class_params = self.node_class_params
        merge = self.input_merge()
        arg_input = self.run_params.get('input', None) if self.run_params is not None else merge
        self.output_data = merge + " " + arg_input
        self.is_finished = True
        return self.output_data


class PlaceholderNode(BaseNode):

    def run(self):
        self.output_data = self.input_merge()
        self.is_finished = True
        return self.output_data
