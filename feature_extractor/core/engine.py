import math
import numpy as np
from monai.metrics import ROCAUCMetric
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import torch
import torch.nn.functional as F

from core.utils import *

auc_metric = ROCAUCMetric() # Average Macro

# Setting...!
fn_tonumpy = lambda x: x.detach().cpu().numpy()


## Only Labeled Data
def train_ENTNet_Labeled(model, data_loader, optimizer, device, epoch, config):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        image  = batch_data[0].to(device).float()
        label  = batch_data[1].to(device).long()
        
        pred = model(image)
        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion(input=pred, target=label)
        else:
            loss = model.criterion(input=pred, target=label)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def valid_ENTNet_Labeled(model, data_loader, device, epoch,  config):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    header = 'Valid: [epoch:{}]'.format(epoch)

    label_list = []
    pred_list = []

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        image = batch_data[0].to(device).float()
        label = batch_data[1].to(device).long()

        pred = model(image)

        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion(input=pred, target=label)
        else:
            loss = model.criterion(input=pred, target=label)

        
        metric_logger.update(loss=loss.item())
 
        # Post-processing
        y_onehot = F.one_hot(label, num_classes=config['num_class'])
        y_pred_prob = F.softmax(pred, dim=1)

        # Metric
        auc = auc_metric(y_pred=y_pred_prob, y=y_onehot) # [B, C]

        # Save
        pred_list.append(fn_tonumpy(pred.argmax(dim=1)).squeeze())
        label_list.append(fn_tonumpy(label).squeeze())
        
    # Metric
    label_list = np.concatenate(label_list, axis=0)
    pred_list  = np.concatenate(pred_list, axis=0)

    auc  = auc_metric.aggregate()
    f1   = f1_score(y_true=label_list, y_pred=pred_list, average='macro')
    acc  = accuracy_score(y_true=label_list, y_pred=pred_list)
    rec  = recall_score(y_true=label_list, y_pred=pred_list, average='macro')
    pre  = precision_score(y_true=label_list, y_pred=pred_list, average='macro')
    metric_logger.update(auc=auc, f1=f1, accuracy=acc, recall=rec, precision=pre)
    
    auc_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


# Soft Pseudo Label
def train_ENTNet_Soft_Pseudo(model, teacher_model, data_loader, optimizer, device, epoch, config):
    model.train(True)
    teacher_model.eval()
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        image  = batch_data[0].to(device).float()
        label  = batch_data[1].to(device).long()
        
        pred = model(image)

        with torch.no_grad():
            teacher_pred = teacher_model(image)

        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion_train(pred, label, teacher_pred)
        else:
            loss = model.criterion_train(pred, label, teacher_pred)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


# Hard Pseudo Label
def train_ENTNet_Hard_Pseudo(model, teacher_model, data_loader, optimizer, device, epoch, config):
    model.train(True)
    teacher_model.eval()
    metric_logger = MetricLogger(delimiter="  ", n=config['batch_size'])
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, config['print_freq'], header):
        image = batch_data[0].to(device).float()
        label = batch_data[1].to(device).long()
        
        pred = model(image)

        with torch.no_grad():
            teacher_pred = teacher_model(image)
        
        teacher_hard_label = teacher_pred.argmax(dim=1)

        if config['gpu_mode'] == 'DataParallel':
            loss = model.module.criterion_train(pred, label, teacher_hard_label)
        else:
            loss = model.criterion_train(pred, label, teacher_hard_label)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}