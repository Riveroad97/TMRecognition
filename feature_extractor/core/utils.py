import os
import time
import random
import datetime
import numpy as np
from collections import defaultdict, deque

import torch


## 네트워크 저장하기
def save(ckpt_dir, model, optimizer, lr_scheduler, epoch, config):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if config['gpu_mode'] == 'Single':
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict()},
                "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
    
    elif config['gpu_mode'] == 'DataParallel':
        torch.save({'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict()},
                "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


## 네트워크 불러오기
def load(ckpt_dir, model, optimizer, lr_scheduler):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return model, optimizer, lr_scheduler, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location='cpu')

    if 'module' in list(dict_model['model'].keys())[0]:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(dict_model['model'], strict=False)
    else:
        model.load_state_dict(dict_model['model'], strict=False)
        
    # model.load_state_dict(dict_model['model'])
    optimizer.load_state_dict(dict_model['optimizer'])
    lr_scheduler.load_state_dict(dict_model['lr_scheduler'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return model, optimizer, lr_scheduler, epoch


## log 읽기
def read_log(path):
    log_list = []
    lines = open(path, 'r').read().splitlines() 
    for i in range(len(lines)):
        exec('log_list.append('+lines[i] + ')')
    return  log_list


def seed_everything(seed):
    random.seed(seed)  # python random seed 고정
    os.environ['PYTHONHASHSEED'] = str(seed)  # os 자체의 seed 고정
    np.random.seed(seed)  # numpy seed 고정
    torch.manual_seed(seed)  # torch seed 고정
    torch.cuda.manual_seed(seed)  # cudnn seed 고정
    torch.backends.cudnn.deterministic = True  # cudnn seed 고정(nn.Conv2d)
    torch.backends.cudnn.benchmark = False  # CUDA 내부 연산에서 가장 빠른 알고리즘을 찾아 수행


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        # n is batch_size
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", n=1):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter  = delimiter
        self.n = n

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(value=v, n=self.n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)


    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
