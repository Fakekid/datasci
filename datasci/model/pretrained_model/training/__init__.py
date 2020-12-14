# coding:utf8

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import *
from tensorboardX import SummaryWriter
import torch.distributed as dist
from ...ModelEvaluation import cls_metrics

opti_dict = {'Adam': Adam, 'SGD': SGD, 'Adadelta': Adadelta, 'Adagrad': Adagrad, 'RMSprop': RMSprop, 'LBFGS': LBFGS}


def get_model_unfolded_outputs(data_batch, model, mode='train', label_field_name='label'):
    """

    Args:
        data_batch:
        model:
        mode:
        label_field_name:

    Returns:

    """
    cudas = torch.cuda.device_count()
    if cudas > 0:
        torch.cuda.empty_cache()

    if isinstance(data_batch, dict):
        items = {var_name: var_value.cuda() if cudas > 0 else var_value for var_name, var_value in
                 data_batch.items()}
        # items = {var_name: var_value for var_name, var_value in data_batch.items()}

        outputs = model(**items)
        logits = outputs.logits
        if mode == 'infer':
            return logits
        elif mode == 'test':
            labels = data_batch[label_field_name]
            return logits, labels

        loss = outputs.loss
        labels = data_batch[label_field_name]
    else:
        items = [t.cuda() if cudas > 0 else t for t in data_batch]
        outputs = model(*items)
        logits, loss = outputs
        labels = data_batch[-1]

    return logits, loss, labels


def downstream_finetune(model, train_data_loader, test_data_loader=None, epoch=16,
                        opti='Adam', opti_params=None, summary_path=None,
                        label_field_name='labels', mask_field_name='attention_mask',
                        model_saved_path=None):
    """

    Args:
        model:
        train_data_loader:
        test_data_loader:
        epoch:
        opti:
        opti_params:
        summary_path:
        label_field_name:
        mask_field_name:

    Returns:

    """
    if opti_params is None:
        opti_params = {}
    is_summary = False
    if summary_path is not None:
        is_summary = True
        summary = SummaryWriter(summary_path)

    cudas = torch.cuda.device_count()

    optimizer = opti_dict.get(opti, Adam)(model.parameters(), **opti_params)

    batch_count = 0
    for ep in range(epoch):
        bar = tqdm(train_data_loader)
        print('Training...')
        # train stage
        model.train()
        for idx, train_batch in enumerate(bar):
            if isinstance(train_batch, dict):
                attention_mask = train_batch[mask_field_name]
            else:
                attention_mask = train_batch[1]

            # clear all weight's gradient
            optimizer.zero_grad()
            # get model's output, including
            logits, loss, labels = get_model_unfolded_outputs(
                train_batch, model, mode='train', label_field_name=label_field_name)

            loss = torch.mean(loss)

            # backward
            loss.backward()
            optimizer.step()

            if bool(cudas):
                labels = labels.cpu().numpy()
                logits = logits.detach().cpu().numpy()
            else:
                labels = labels.numpy()
                logits = logits.detach().numpy()

            metrics_dic = cls_metrics(labels, logits, mask=attention_mask, data_on_gpu=bool(cudas))
            acc, p, r, auc, f1 = \
                metrics_dic['acc'], metrics_dic['p'], metrics_dic['r'], metrics_dic['auc'], metrics_dic['f1']

            batch_count = ep * len(train_data_loader) + idx
            if is_summary:
                summary.add_scalar('train_batch_loss', np.mean(loss.detach().cpu().numpy()), batch_count)
                summary.add_scalar('train_batch_acc', acc, batch_count)
                summary.add_scalar('train_batch_p', p, batch_count)
                summary.add_scalar('train_batch_r', r, batch_count)
                summary.add_scalar('train_batch_auc', auc, batch_count)
                summary.add_scalar('train_batch_f1', f1, batch_count)

            bar.set_description(
                'e:{}, step:{}, loss:{}, acc:{}, p:{}, r:{}, auc:{}, f1:{}'
                    .format(ep, batch_count, round(loss.detach().item(), 3), round(acc, 3), round(p, 3), round(r, 3),
                            round(auc, 3), round(f1, 3)))

        print('Testing...')
        # test stage
        model.eval()
        accu_acc = 0
        accu_p = 0
        accu_r = 0
        accu_auc = 0
        accu_f1 = 0
        bar = tqdm(test_data_loader)
        for idx, test_batch in enumerate(bar):
            if isinstance(test_batch, dict):
                attention_mask = test_batch[mask_field_name]
            else:
                attention_mask = test_batch[1]

            with torch.no_grad():
                logits, labels = get_model_unfolded_outputs(
                    test_batch, model, mode='test', label_field_name=label_field_name)
                # test_batch_acc = acc(labels=labels, logits=logits, mask=attention_mask, data_on_gpu=bool(cudas))
                if bool(cudas):
                    labels = labels.cpu().numpy()
                    logits = logits.detach().cpu().numpy()
                else:
                    labels = labels.numpy()
                    logits = logits.detach().numpy()

                metrics_dic = cls_metrics(labels, logits, mask=attention_mask, data_on_gpu=bool(cudas))
                acc, p, r, auc, f1 = \
                    metrics_dic['acc'], metrics_dic['p'], metrics_dic['r'], metrics_dic['auc'], metrics_dic['f1']
                accu_acc += acc
                accu_p += p
                accu_r += r
                accu_auc += auc
                accu_f1 += f1
                bar.set_description(
                    'epoch:{}, step:{}, batch acc:{}, batch acc:{}, batch acc:{}, batch acc:{}, batch acc:{}'
                        .format(ep, batch_count, acc, p, r, auc, f1))

        accu_acc /= len(test_data_loader)
        accu_p /= len(test_data_loader)
        accu_r /= len(test_data_loader)
        accu_auc /= len(test_data_loader)
        accu_f1 /= len(test_data_loader)
        if is_summary:
            summary.add_scalar('test_batch_acc', accu_acc, batch_count)
            summary.add_scalar('test_batch_p', accu_p, batch_count)
            summary.add_scalar('test_batch_r', accu_r, batch_count)
            summary.add_scalar('test_batch_auc', accu_auc, batch_count)
            summary.add_scalar('test_batch_f1', accu_f1, batch_count)

        if model_saved_path is not None:
            if not os.path.exists(model_saved_path[:model_saved_path.rfind('/')]):
                os.mkdir(model_saved_path[:model_saved_path.rfind('/')])
            torch.save(model, '{}-ckpt-{}'.format(model_saved_path, batch_count))
            print('saved checkpoint done.')
