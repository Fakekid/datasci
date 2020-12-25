import logging
import os
from datasci.utils.mylog import get_file_logger, get_stream_logger
from datasci.utils.path_check import check_path

log = get_stream_logger("global_config", level=logging.INFO)

PROJECT_PATH = os.getcwd()
config_dir = os.path.join(PROJECT_PATH, "conf")
check_path(config_dir)
log_dir = os.path.join(PROJECT_PATH, "log")
check_path(log_dir)
data_dir = os.path.join(PROJECT_PATH, "data")
check_path(data_dir)
global_config = {
    "db_config": os.path.join(config_dir, "db_config.ini"),
    "ticket": os.path.join(config_dir, "ticket"),
    "encoder": os.path.join(config_dir, "encoder_config.json"),
    "model": os.path.join(config_dir, "model_config.json"),
    "strategy": os.path.join(config_dir, "strategy_config.json"),
    "feature": os.path.join(config_dir, "feature_config.json"),
    "job": os.path.join(config_dir, "jobs_config.json"),
    "project_path": PROJECT_PATH,
    "log_path": log_dir,
    "data_path": data_dir
}
log.debug("Global config : %s" % global_config)
