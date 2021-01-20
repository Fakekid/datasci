import json
import time
from queue import Queue

from datasci.workflow.node import EndNode

from datasci.utils.reflection import Reflection
import pandas as pd


def _init_node(nodename, config):
    node_class = config.get(nodename).get("node_class")
    idx = node_class.rfind(".")
    module_path = node_class[0: idx]
    class_name = node_class[idx + 1: len(node_class)]
    if module_path is None:
        print("Module path is None!")
        exit(-1)
    if class_name is None:
        print("Class name is None!")
        exit(-1)

    params = config.get(nodename).get("params", None)
    if len(params) == 0:
        params = None

    node_class_params = None
    run_params = None
    if params is not None:
        node_class_params = params.get("node_class_params", None)
        run_params = params.get("run_params", None)

    next_nodes = config.get(nodename).get("next", None)
    if len(next_nodes) == 0 or next_nodes == "" or next_nodes == []:
        next_nodes = None
    input_data_file = config.get(nodename).get("input")
    if input_data_file is None or input_data_file == "":
        input_data = None
    else:
        input_data = pd.read_csv(input_data_file)
    init_params = {
        "node_name": nodename,
        "next_nodes": next_nodes,
        "node_class_params": node_class_params,
        "run_params": run_params,
        "input_data": input_data
    }
    cls_obj = Reflection.reflect_obj(module_path=module_path, class_name=class_name, params=init_params)

    return cls_obj


def run(config=None):

    logo = '''

    ________         _____         ________       _____        ___       __               ______  ________________                  
    ___  __ \______ ___  /_______ ___  ___/__________(_)       __ |     / /______ ___________  /_____  ____/___  /______ ___      __
    __  / / /_  __ `/_  __/_  __ `/_____ \ _  ___/__  /        __ | /| / / _  __ \__  ___/__  //_/__  /_    __  / _  __ \__ | /| / /
    _  /_/ / / /_/ / / /_  / /_/ / ____/ / / /__  _  /         __ |/ |/ /  / /_/ /_  /    _  ,<   _  __/    _  /  / /_/ /__ |/ |/ / 
    /_____/  \__,_/  \__/  \__,_/  /____/  \___/  /_/          ____/|__/   \____/ /_/     /_/|_|  /_/       /_/   \____/ ____/|__/  


            '''
    print(logo)

    result = {}
    if config is None:
        from datasci.workflow.config.global_config import global_config
        config = global_config.get("workflow")
    with open(config) as f:
        conf = f.read()
    run_dag_config = json.loads(conf)

    q = Queue(maxsize=0)

    start_node = _init_node(nodename='start', config=run_dag_config)
    q.put(start_node)
    i = 1
    while q.qsize() != 0:
        node = q.get()
        print("------------------------ STEP %s ------------------------" % i)
        print("NODE NAME : %s, START TIME : %s" % (
        node.node_name, time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
        print(">>> %s Node Running ... ... " % node.node_name)
        ret = node.run()
        if isinstance(node, EndNode):
            result[node.node_name] = ret
        print(">>> %s Node Finished ! " % node.node_name)
        print("NODE NAME : %s, FINISHED TIME : %s " % (
        node.node_name, time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
        print("\n")
        i += 1
        if node.next_nodes is not None:
            for n_name in node.next_nodes:
                sub_node = _init_node(nodename=n_name, config=run_dag_config)
                if sub_node.input_data is None:
                    sub_node.input_data = node.output_data
                q.put(sub_node)
        else:
            continue
    print("------------------------ FINISHED ------------------------")
    return result
