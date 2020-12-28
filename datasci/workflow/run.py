import json
import time
from queue import Queue
from datasci.workflow.node import Node
import pandas as pd


def _init_node(nodename, config):
    node_type = config.get(nodename).get("node_type")
    next_nodes = config.get(nodename).get("next")
    if len(next_nodes) == 0 or next_nodes == "" or next_nodes == []:
        next_nodes = None
    runable = config.get(nodename).get("runable")
    params = config.get(nodename).get("params")
    input_data_file = config.get(nodename).get("input")
    if input_data_file is None or input_data_file == "":
        input_data = None
    else:
        input_data = pd.read_csv(input_data_file)
    if len(params) == 0:
        params = None
    node = Node(node_name=nodename, node_type=node_type, next_nodes=next_nodes, runable=runable, params=params,
                input_data=input_data)
    return node


def run(config=None):
    if config is None:
        from datasci.workflow.config.global_config import global_config
        config = global_config.get("workflow")
    with open(config) as f:
        conf = f.read()
    run_dag_config = json.loads(conf)

    q = Queue(maxsize=0)

    start_node = _init_node(nodename='start', config=run_dag_config)
    q.put(start_node)
    ret = None
    i = 1
    while q.qsize() != 0:
        node = q.get()
        print("----------------------------------------")
        print("STEP %s START : %s node is running ...." % (i, node.node_name))
        print("... ...")
        ret = node.run()
        print("STEP %s FINISHED : %s node is finished...." % (i, node.node_name))
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
    return ret
