from datasci.utils.mysql_utils import MysqlUtils
from datasci.workflow.node.base import BaseNode
import json
from datetime import datetime, date, timedelta


def get_bias_ratio(value, base_value):
    bias_ratio = round(abs((base_value - value) / (base_value + 1e-7)), 4) if base_value is not None else 0.0
    return bias_ratio

class MonitorLogNode(BaseNode):

    def run(self):
        if self.input_data is not None:
            self.input_data = self.input_merge(axis=0)
        section = 'Mysql-data_bank'
        mysql_Utils = MysqlUtils(section)
        min_timedelta = self.run_params.get('min_timedelta', None) if self.run_params is not None else -7
        max_timedelta = self.run_params.get('max_timedelta', None) if self.run_params is not None else 0

        min_time_tag = (date.today() + timedelta(days=min_timedelta)).strftime("%Y%m%d")
        max_time_tag = (date.today() + timedelta(days=max_timedelta)).strftime("%Y%m%d")

        sql = """
            select
                   records.model_id,
                   records.model_name,
                   records.product_line,
                   records.task_name,
                   records.time_tag,
                   records.indicator_value,
                   configs.action,
                   configs.status,
                   configs.metric_value
            from
                 ( select * from xes_1v1_model_monitor_records where time_tag >= {0} and time_tag <= {1}) records
            left join
                     xes_1v1_model_monitor_configs configs
            on records.model_id = configs.model_id
            order by records.time_tag desc
        """.format(min_time_tag, max_time_tag)

        result = mysql_Utils.get_result_sql(sql=sql)
        format_result = dict()
        for ret in result:
            model_id = str(ret[0])
            time_tag = ret[4]
            if model_id in format_result:
                format_result[model_id][time_tag] = ret
            else:
                format_result[model_id] = dict()
                format_result[model_id][time_tag] = ret
        data_info = list()
        for model_id, data in format_result.items():
            date_tags = list(data.keys())
            date_tags.sort(reverse=True)
            update_date_tag = date_tags[0]
            indicator_values_1d = dict()
            indicator_values_2d = dict()
            indicator_values_3d = dict()
            indicator_values_5d = dict()
            indicator_values_7d = dict()
            if len(date_tags) >= 2:
                date_tags_1d = date_tags[1]
                indicator_values_1d = json.loads(data[date_tags_1d][5])
            if len(date_tags) >= 3:
                date_tags_2d = date_tags[2]
                indicator_values_2d = json.loads(data[date_tags_2d][5])
            if len(date_tags) >= 4:
                date_tags_3d = date_tags[3]
                indicator_values_3d = json.loads(data[date_tags_3d][5])
            if len(date_tags) >= 6:
                date_tags_5d = date_tags[5]
                indicator_values_5d = json.loads(data[date_tags_5d][5])
            if len(date_tags) >= 8:
                date_tags_7d = date_tags[7]
                indicator_values_7d = json.loads(data[date_tags_7d][5])
            update_data = data[update_date_tag]
            model_name = update_data[1]
            product_line = update_data[2]
            task_name = update_data[3]
            time_tag = update_data[4]
            indicator_values = json.loads(update_data[5])
            action = update_data[6]
            status = update_data[7]
            metric_values = json.loads(update_data[8])

            for indicator, value in indicator_values.items():
                bias_ratio = get_bias_ratio(value, metric_values.get(indicator, None))
                attenuation_ratio_1d = get_bias_ratio(value, indicator_values_1d.get(indicator, None))
                attenuation_ratio_2d = get_bias_ratio(value, indicator_values_2d.get(indicator, None))
                attenuation_ratio_3d = get_bias_ratio(value, indicator_values_3d.get(indicator, None))
                attenuation_ratio_5d = get_bias_ratio(value, indicator_values_5d.get(indicator, None))
                attenuation_ratio_7d = get_bias_ratio(value, indicator_values_7d.get(indicator, None))
                data_info.append((model_id, model_name, product_line, task_name, indicator,
                                  metric_values.get(indicator, None), value, bias_ratio
                                  , attenuation_ratio_1d, attenuation_ratio_2d, attenuation_ratio_3d,
                                  attenuation_ratio_5d, attenuation_ratio_7d, status, action, time_tag))
        s = ['%s'] * 16
        insert_sql = "insert into xes_1v1_model_monitor_logs " \
                     "(model_id, model_name, product_line, task_name, indicator,base_value, value, bias_ratio, attenuation_ratio_1d, attenuation_ratio_2d, attenuation_ratio_3d, attenuation_ratio_5d,attenuation_ratio_7d, status, action, time_tag)" \
                     "VALUES ( " + ",".join(s) + ")"
        mysql_Utils.get_executemany_sql(sql=insert_sql, data_info=data_info)

        self.output_data = None
        self.is_finished = True
        return self.output_data


class AlertMsgNode(BaseNode):

    def run(self):
        if self.input_data is not None:
            self.input_data = self.input_merge(axis=0)
        section = 'Mysql-data_bank'
        mysql_Utils = MysqlUtils(section)
        timedelay = self.run_params.get('timedelta', None) if self.run_params is not None else 0
        max_bias = self.run_params.get('max_bias', None) if self.run_params is not None else 0.1
        time_tag = (date.today() + timedelta(days=timedelay)).strftime("%Y%m%d")
        sql = """
            select
                distinct
                model_id,
                model_name,
                product_line,
                indicator,
                base_value,
                value,
                bias_ratio,
                attenuation_ratio_1d,
                attenuation_ratio_2d,
                attenuation_ratio_3d,
                attenuation_ratio_5d,
                attenuation_ratio_7d,
                status,
                action
            from
                xes_1v1_model_monitor_logs
            where
                time_tag = {0} 
                and (bias_ratio > {1}
                or attenuation_ratio_1d > {1} 
                or attenuation_ratio_2d > {1}
                or attenuation_ratio_3d > {1}
                or attenuation_ratio_5d > {1}
                or attenuation_ratio_7d > {1}
                ) 
        """.format(time_tag, max_bias)
        result = mysql_Utils.get_result_sql(sql=sql)
        alert_msg = None
        i = 1
        for ret in result:
            status = ret[12]
            action = ret[13]
            if status == 1 and action == 1:
                msg = "告警 %s：模型id为 %s (模型名称 %s, 业务线 %s）， 指标 %s， 基准值 %s，评估值 %s，" \
                  "基准偏移率 %s， 1天衰减率 %s， 2天衰减率 %s ，3天衰减率 %s ，5天衰减率 %s， 7天衰减率 %s \n" \
                  % (i, ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], ret[6], ret[7], ret[8], ret[9], ret[10], ret[11])
                alert_msg = alert_msg + msg if alert_msg is not None else msg
                i = i + 1
        self.output_data = alert_msg
        self.is_finished = True
        return self.output_data



