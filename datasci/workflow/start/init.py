from datasci.workflow.config.global_config import *
from datasci.workflow.config.task_config import get_config
from datasci.utils.path_check import check_path
from datasci.utils.mylog import get_stream_logger
import os


class InitProcesser(object):
    def __init__(self, config=None, fconfig=None, encoder_map=None, model_map=None, log=None):
        """
            A packaging of train

            Args
            -------
            config
                Job config dict , which like the content of "conf/job_config.json"
            config_file
                Job config file path ,.eg "conf/job_config.json"
            fconfig
                Feature config, which like the content of "conf/feature_config.json"
            fconfig_file
                Feature config,e.g. "conf/feature_config.json"

            encoder_map
                encoder map , which like the content of "conf/encoder_config.json"
            encoder_map_file
                encoder map,e.g. "conf/encoder_config.json"

            model_map
                model map, which like the content of "conf/model_config.json"
            model_map_file
                model map,,e.g. "conf/model_config.json"

            Returns
            -------
            None
        """
        from datasci.workflow.config.log_config import log_level
        self.log = get_stream_logger("INIT", level=log_level) if log is None else log
        # self.jobs = get_config(config_type="job", config=config)
        # self.log.debug("Job config is : %s" % self.jobs)
        # self.encoders = get_config(config_type="encoder", config=encoder_map)
        # self.log.debug("Encoder config is : %s" % self.jobs)
        # self.features = get_config(config_type="feature", config=fconfig)
        # self.log.debug("Feature config is : %s" % self.features)
        # self.models = get_config(config_type="model", config=model_map)
        # self.log.debug("Model config is : %s" % self.models)
        # paths = self.jobs.get('paths')
        # self.project_path = paths.get('project_path')
        # self.data_path = paths.get('data_path')
        # self.result_path = paths.get('result_path')
        # self.pre_model_path = paths.get('pre_model_path')
        # self.model_path = paths.get('model_path')
        # self.feature_process_path = paths.get('feature_package_path')
        # self.train_data_path = paths.get('train_data_path')
        # self.predict_data_path = paths.get('predict_data_path')

    def run(self):
        # check paths
        check_path(PROJECT_PATH, is_make=True)
        self.log.info("Project path is : %s" % PROJECT_PATH)
        check_path(config_dir, is_make=True)
        self.log.info("Config data path is : %s" % config_dir)
        check_path(log_dir, is_make=True)
        self.log.info("Log data path is : %s" % log_dir)
        check_path(data_dir, is_make=True)

        # check sub paths
        # self.data_path = os.path.join(PROJECT_PATH, self.data_path) if not os.path.isabs(
        #     self.data_path) else self.data_path
        # if PROJECT_PATH != data_dir:
        #     check_path(self.data_path, is_make=True)
        # self.log.info("Data path is : %s" % self.data_path)

        # self.result_path = os.path.join(self.data_path, self.result_path) if not os.path.isabs(
        #     self.result_path) else self.result_path
        # check_path(self.result_path, is_make=True)
        # self.log.info("Result path is : %s" % self.result_path)
        #
        # self.pre_model_path = os.path.join(self.data_path, self.pre_model_path) if not os.path.isabs(
        #     self.pre_model_path) else self.pre_model_path
        # check_path(self.pre_model_path, is_make=True)
        # self.log.info("Pre Model path is : %s" % self.pre_model_path)
        #
        # self.model_path = os.path.join(self.data_path, self.model_path) if not os.path.isabs(
        #     self.model_path) else self.model_path
        # check_path(self.model_path, is_make=True)
        # self.log.info("Model path is : %s" % self.model_path)
        #
        # self.feature_process_path = os.path.join(self.data_path, self.feature_process_path) if not os.path.isabs(
        #     self.feature_process_path) else self.feature_process_path
        # check_path(self.feature_process_path, is_make=True)
        # self.log.info("Feature path is : %s" % self.feature_process_path)
        #
        # self.train_data_path = os.path.join(self.data_path, self.train_data_path) if not os.path.isabs(
        #     self.train_data_path) else self.train_data_path
        # check_path(self.train_data_path, is_make=True)
        # self.log.info("Train data path is : %s" % self.train_data_path)
        #
        # self.predict_data_path = os.path.join(self.data_path, self.predict_data_path) if not os.path.isabs(
        #     self.predict_data_path) else self.predict_data_path
        # check_path(self.predict_data_path, is_make=True)
        # self.log.info("Predict data path is : %s" % self.predict_data_path)
