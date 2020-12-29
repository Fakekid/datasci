import time

from datasci.workflow.predict.predict import PredictProcesser
from datasci.workflow.train.train import TrainProcesser


class TrainNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.params = params

    def run(self, multi_process=True):
        if self.params is None:
            train_class = TrainProcesser()
        else:
            train_class = TrainProcesser(**self.params)
        train_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = None
        self.is_finished = True
        return self.output_data


class PredictNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.params = params

    def run(self, multi_process=True):
        if self.params is None:
            predict_class = PredictProcesser()
        else:
            predict_class = PredictProcesser(**self.params)
        result = predict_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = result
        self.is_finished = True
        return self.output_data


class PredictAndJoinNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.params = params

    def run(self, multi_process=True):
        if self.params is None:
            predict_class = PredictProcesser()
        else:
            predict_class = PredictProcesser(**self.params)
        result = predict_class.run(data=self.input_data, multi_process=multi_process)
        result['join'] = predict_class.join(result)
        self.output_data = result
        self.is_finished = True
        return self.output_data


class PredictAndSaveNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.params = params

    def run(self, multi_process=True):
        if self.params is None:
            predict_class = PredictProcesser()
        else:
            predict_class = PredictProcesser(**self.params)
        result = predict_class.run(data=self.input_data, multi_process=multi_process)
        result['join'] = predict_class.join(result)
        ex_col = {
            'model_version': 'join',
            'dt': "%s" % time.strftime("%Y%m%d", time.localtime())
        }
        predict_class.save(data=result.get('join'), data_tag='join', extend_columns=ex_col)
        self.output_data = result
        self.is_finished = True
        return self.output_data


class BeginNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.params = params

    def run(self, multi_process=False):
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class EndNode(object):
    def __init__(self, node_name, next_nodes, input_data=None, params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.params = params

    def run(self, multi_process=False):
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data
