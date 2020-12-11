# coding:utf8

import torch
from transformers import *
from torch import nn

model_lib = {
    'cls': BertForSequenceClassification,
    'tag': BertForTokenClassification
}


def bert_downstream_model(pretrained_model, num_cls, model_type='cls', return_dict=True):
    """

    Args:
        pretrained_model: str value, pretrained model's name or local directory
        num_cls: int value, number of categories
        model_type: str value, downstream task type, default 'cls', current only support 'cls'、'tag'
        return_dict: bool value, whether return dict type value, default True

    Returns:
        bert model
        bert tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model_class = model_lib.get(model_type)
    model = model_class.from_pretrained(pretrained_model, num_labels=num_cls, return_dict=return_dict)
    cudas = torch.cuda.device_count()
    if cudas == 0:
        print('load bert cls model to CPU')
        return model, tokenizer
    else:
        if cudas >= 1:
            print('load bert cls model to GPU')
            model = nn.DataParallel(model)
        return model.cuda(), tokenizer
