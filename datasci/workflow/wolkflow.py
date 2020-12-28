# -*- coding:utf-8 -*-
import json
import sys
import time
import getopt

from datasci.workflow.predict.predict import PredictProcesser
from datasci.workflow.train.train import TrainProcesser
import pandas as pd
from datasci.utils.mylog import get_file_logger, get_stream_logger

log = get_stream_logger("workflow")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

def main(argv):
    job_type = None
    job_config = None
    feature_config = None
    encoder_config = None
    model_config = None
    multi_process = False
    batch_size = 10000
    join_key = None
    try:
        opts, args = getopt.getopt(argv, "j:J:f:e:m:hpb:k:",
                                   ["job-type=", "job-config=", "feature-config=", "encoder-config=", "model-config=",
                                    "help", "multi-process", "batch-size=", "join-key="])
    except getopt.GetoptError as e:
        log.error(e)
        log.error(
            'Usage: python  worlflow.py -j <job-type> -J <job-config> -f <feature-config> -e <encoder-config> -m <model-config> -p')
        sys.exit(-1)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(
                """
                -----------------------------------------------------HELP-----------------------------------------------------

                worlflow.py -j <job-type> -J <job-config> -f <feature-config> -e <encoder-config> -m <model-config> -p

                -j, --job-type : job type train|predict|output
                -J, --job-config : job config. python dict or file , default file which is  conf/job_config.json
                -f, --feature-config : feature config. Json string or file , default file which is  conf/feature_config.json
                -e, --encoder-config : encoder config. Json string or file , default file which is  conf/encoder_config.json
                -m, --model-config : model config. python dict or file , default file which is  conf/model_config.json
                -p, --multi-process : multi process mode
                -b, --batch-size : predict result write batch size
                -k, --join-key : predict result join on this key 
            """
            )
            sys.exit(0)
        elif opt in ("-j", "--job-type"):
            job_type = arg
        elif opt in ("-J", "--job-config"):
            job_config = arg
        elif opt in ("-f", "--feature-config"):
            feature_config = arg
        elif opt in ("-e", "--encoder-config"):
            encoder_config = arg
        elif opt in ("-m", "--model-config"):
            model_config = arg
        elif opt in ("-p", "--multi-process"):
            multi_process = True
        elif opt in ("-b", "--batch-size"):
            batch_size = arg
        elif opt in ("-k", "--join-key"):
            join_key = arg

    if job_type is None:
        job_type = "predict"

    log.info("Job type is %s" % job_type)

    if job_type == 'train':
        train_class = TrainProcesser(config=job_config, fconfig=feature_config, encoder_map=encoder_config,
                                     model_map=model_config)
        train_class.run(multi_process=multi_process)

    if job_type == 'predict':
        predict_class = PredictProcesser(config=job_config, model_map=model_config)
        result = predict_class.run(multi_process=multi_process)
        result['join'] = predict_class.join(result, join_key=join_key)
        ex_col = {
            'model_version': 'join',
            'dt': "%s" % time.strftime("%Y%m%d", time.localtime())
        }
        predict_class.save(data=result.get('join'), data_tag='join', extend_columns=ex_col, pagesize=batch_size)


if __name__ == '__main__':
    main(sys.argv[1:])
