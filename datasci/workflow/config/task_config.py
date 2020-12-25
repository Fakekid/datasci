import json

def _get_file_config(type, config):
    config_file = config
    if config_file is None:
        from datasci.workflow.config.global_config import global_config
        config_file = global_config.get(type)
    if config_file is not None:
        with open(config_file) as f:
            conf = f.read()
            return json.loads(conf)

def get_config(type, config, file_config=None):
    conf =  _get_file_config(type=type, config=file_config) if config is None else config
    return conf
