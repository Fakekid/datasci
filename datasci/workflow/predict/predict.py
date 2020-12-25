# -*- coding:utf-8 -*-
import logging
import os
from collections import Iterable
import time
import numpy as np
import threading

from datasci.workflow.feature.feature_process import GroupFeatureProcesser
from datasci.workflow.predict.predict_package import PredictPackage
from datasci.workflow.input.get_data import get_data
from datasci.workflow.output.save_data import save_data
import pandas as pd
from datasci.utils.mylog import get_stream_logger
from datasci.utils.path_check import check_path
from datasci.workflow.config.task_config import get_config

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

log = get_stream_logger('PredictProcesser', level=logging.INFO)
threadLock = threading.Lock()
threads = []

class MultiPredictThread(threading.Thread):
    def __init__(self, thread_id, processer, model_name, model_config, result_dict, data=None):
        threading.Thread.__init__(self)
        self.log = get_stream_logger('MultiPredictThread: %s' % thread_id)
        self.processer = processer
        self.model_name = model_name
        self.model_config = model_config
        self.data = data
        self.result_dict = result_dict

    def run(self):
        threadLock.acquire()
        self.log.info('The thread of %s starting ... ...' % self.model_name)
        threadLock.release()
        self.processer._run(model_name=self.model_name, model_config=self.model_config,
                            result_dict=self.result_dict, data=self.data)
        threadLock.acquire()
        self.log.info('The thread of %s finished' % self.model_name)
        threadLock.release()

class PredictProcesser(object):

    def __init__(self, config=None, model_map=None, multi_process=False):
        """
            A packaging of predict process

            Args
            -------
            config
                Job config dict , which like the content of "conf/job_config.json"
            config_file
                Job config file path ,e.g. "conf/job_config.json"
            Returns
            -------
            None
        """
        self.jobs = get_config(config_type="job", config=config)
        log.debug("Job config is : %s" % self.jobs)
        self.models = get_config(config_type="model", config=model_map)
        log.debug("Model config is : %s" % self.models)
        self.output = self.jobs.get('output')
        self.join_key = self.output.get('join_key')

        paths = self.jobs.get('paths')
        self.project_path = paths.get('project_path')
        check_path(self.project_path)
        self.result_path = paths.get('result_path')
        check_path(self.result_path)
        self.pre_model_path = paths.get('pre_model_path')
        check_path(self.pre_model_path)
        self.model_path = paths.get('model_path')
        check_path(self.model_path)
        self.feature_process_path = paths.get('feature_package_path')
        check_path(self.feature_process_path)
        self.multi_process = multi_process

    def run(self, data=None):
        models_config = self.jobs.get('models')
        result_dict = dict()
        if self.multi_process:
            i = 0
            for model_name, model_config in models_config.items():
                new_thread = MultiPredictThread(thread_id=i, processer=self, model_name=model_name, model_config=model_config,result_dict=result_dict,data=data)
                new_thread.start()
                threads.append(new_thread)
                i = i + 1
            for t in threads:
                t.join()
            return result_dict
        else:
            for model_name, model_config in models_config.items():
                log.info('The process of %s starting ... ...' % model_name)
                self._run(model_name=model_name, model_config=model_config, result_dict=result_dict, data=data)
                log.info('The process of %s finished' % model_name)
            return result_dict

    def _run(self, model_name, model_config, result_dict, data=None):
        is_online = model_config.get('predict').get('is_online', False)
        if is_online:
            model_type = model_config.get('model_type', None)
            is_save = model_config.get('predict').get('is_save', False)

            feature_process_file = model_config.get('predict').get('feature_package_file', None)
            sub_feature_process_path = os.path.join(self.feature_process_path, model_name)
            full_feature_process_file = os.path.join(sub_feature_process_path, feature_process_file)
            if not os.path.exists(full_feature_process_file):
                log.error("Feature group process file %s is not exists ! " % full_feature_process_file)
                exit(-1)
            feature_process = GroupFeatureProcesser.read_feature_processer(full_feature_process_file)

            model_file = model_config.get('predict').get('model_file', None)
            sub_model_path = os.path.join(self.model_path, model_name)
            full_model_file = os.path.join(sub_model_path, model_file)

            input_config = model_config.get('input').get('predict_data')
            output_config = model_config.get('output')

            model_version = model_config.get('model_version')
            predict_package = PredictPackage(
                model_name=model_name,
                model_version=model_version,
                model_type=model_type,
                model_file=full_model_file,
                model_map=self.models,

            )
            ret = self.predict(predict_package=predict_package, feature_process=feature_process,
                               input_config=input_config, data=data)
            if is_save:
                ex_col = {
                    'model_version': model_version,
                    'dt': "%s" % time.strftime("%Y%m%d", time.localtime())
                }
                self.save(data=ret, output_config=output_config, extend_columns=ex_col)
            result_dict[model_name] = ret
        return result_dict

    def join(self, data_dict, join_key=None):
        """
        :param data_dict: data dict
        :param join_key:  join  key
        :return: join data
        """
        join_key = self.join_key if join_key is None else join_key
        result = None
        presize = 0
        if join_key is not None:
            for key, value in data_dict.items():
                if join_key in value.columns.tolist():
                    size = value.shape[0]
                    if result is None:
                        presize = size
                        result = value
                    elif presize == size:
                        result = pd.merge(result, value, on=join_key)
        return result

    def save(self, data, output_config=None, data_tag=None, extend_columns=None, pagesize=10000):
        """

        :param data: pd.Datafreame
        :param output_config:  like

            {
                "join":  # data_tag
                {
                    "data_type": "doris",
                    "file": {
                        "path": "/Users/zhaolihan/Documents/TAL/git/xes1v1_data_science/user_intention/data_source/datafile/predict/predict_data_0.csv"
                    },
                    "mysql": {
                        "table": "xes_1v1_user_intention_model_result_test"
                    },
                    "doris": {
                        "table": "ads_user_intention_join_model_predict_result"
                    }
                }
            }
        :param data_tag: like 'join' in the json string of output_config
        :param extend_columns: extend columns in a dict
        :param pagesize: how much lines to write in one step
        :return:
        """

        output_config = self.output.get(data_tag) if output_config is None else output_config
        size = data.shape[0]
        batch_num = size // pagesize + 1
        for i in range(batch_num):
            begin = i * pagesize
            end = begin + pagesize if begin + pagesize < size else size
            batch_data = data.iloc[begin: end]
            save_d = batch_data.copy()
            if extend_columns is not None:
                for col_name, col_value in extend_columns.items():
                    save_d[col_name] = col_value
            log.info('Save data which batch number is %s ' % i)
            save_data(save_d, output_config)

    def predict(self, predict_package, feature_process, data=None, input_config=None):
        """
        Predict data and save result, get data from config
        Args
            -------
            predict_package
                An instance of PredictPackage

            feature_process
                An instance of FeaturePrcoesser

            data
                predict data

            input_config
                input config

            join key

        Returns
        -------
        result
            result and save
        """
        if data is not None:
            tdata = data
        else:
            tdata = get_data(input_config)
        result = pd.DataFrame()
        if isinstance(tdata, Iterable) and not isinstance(tdata, pd.DataFrame):
            for data in tdata:
                data[data.isnull()] = np.NaN
                data.drop_duplicates(inplace=True)
                join_data = None
                if self.join_key in tdata.columns.tolist():
                    join_data = tdata[self.join_key]

                log.info('%s feature engineering starting ... ...' %  predict_package.model_name)
                select_data = feature_process.select_columns(data=data)
                feature_package = feature_process.get_feature_package()
                predict_data = feature_package.transform(select_data)

                rst_data = predict_package.predict(predict_data)
                try:
                    class_num = rst_data.shape[1]
                except IndexError:
                    class_num = 1
                result_col = ['%s_%s' % (predict_package.model_name, i) for i in range(class_num)]
                if join_data is not None:
                    ret = pd.concat((join_data, pd.DataFrame(rst_data, columns=result_col)), axis=1)
                else:
                    ret = pd.DataFrame(rst_data, columns=result_col)
                if result.empty:
                    result = ret
                else:
                    result = pd.concat((result, ret), axis=0)
        else:
            tdata[tdata.isnull()] = np.NaN
            tdata.drop_duplicates(inplace=True)
            join_data = None
            if self.join_key in tdata.columns.tolist():
                join_data = tdata[self.join_key]

            log.info('%s feature engineering starting ... ...' % predict_package.model_name)
            select_data = feature_process.select_columns(data=tdata)
            feature_package = feature_process.get_feature_package()
            predict_data = feature_package.transform(select_data)

            ret = predict_package.predict(predict_data)
            try:
                class_num = ret.shape[1]
            except IndexError:
                class_num = 1
            result_col = ['%s_%s' % (predict_package.model_name, i) for i in range(class_num)]
            if join_data is not None:
                result = pd.concat((join_data, pd.DataFrame(ret, columns=result_col)), axis=1)
            else:
                result = pd.dataframe(ret, columns=result_col)
        return result
