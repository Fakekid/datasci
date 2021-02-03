from datasci.workflow.node.base import BaseNode
from datasci.workflow.train.train import TrainProcesser
from datasci.workflow.predict.predict import PredictProcesser


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
