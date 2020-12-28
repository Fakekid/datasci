import time

from datasci.workflow.predict.predict import PredictProcesser
from datasci.workflow.train.train import TrainProcesser

class Node(object):
    def __init__(self, node_name, node_type, next_nodes, runable, input_data=None, output_data=None, is_finished=False, params=None):
        self.node_name = node_name
        self.node_type = node_type
        self.next_nodes = next_nodes
        self.input_data = input_data
        self.output_data = output_data
        self.runable = runable
        self.is_finished = is_finished
        self.params = params

    def run(self):
        if self.runable:
            if self.node_type == "train" :
                if self.params is None:
                    train_class = TrainProcesser()
                else:
                    train_class = TrainProcesser(**self.params)
                train_class.run(data=self.input_data, multi_process=True)
                self.output_data = None
                self.is_finished = True

            elif self.node_type == "predict":
                if self.params is None:
                    predict_class = PredictProcesser()
                else:
                    predict_class = PredictProcesser(**self.params)
                result = predict_class.run(data=self.input_data, multi_process=True)
                self.output_data = result
                self.is_finished = True

            elif self.node_type == "save":
                if self.params is None:
                    predict_class = PredictProcesser()
                else:
                    predict_class = PredictProcesser(**self.params)
                result = predict_class.run(data=self.input_data, multi_process=True)
                result['join'] = predict_class.join(result)
                ex_col = {
                    'model_version': 'join',
                    'dt': "%s" % time.strftime("%Y%m%d", time.localtime())
                }
                predict_class.save(data=result.get('join'), data_tag='join', extend_columns=ex_col)
                self.output_data = result
                self.is_finished = True
            else:
                self.output_data = self.input_data
                self.is_finished = True
        else:
            self.output_data = self.input_data
            self.is_finished = True
        return self.output_data
