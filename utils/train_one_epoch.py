import torch
import math
import sys
from typing import Iterable
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, epoch_step):
    model.train(True)
    acc1_meter = AverageMeter()
    loss_all = 0

    with tqdm(total=epoch_step, desc=f'Train: ', postfix=dict, mininterval=0.3) as pbar:
        for data_iter_step, batch in enumerate(data_loader):
            rgb = batch[0]
            depth = batch[1]
            target = batch[-1]
            rgb = rgb.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            targets = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(rgb, depth)

            loss = criterion(outputs, targets)
            
            loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1))
            loss_value = loss.item()
            loss_all += loss_value

            if not math.isfinite(loss_value):
                print('Loss is {}, stopping training'.format(loss_value))
                sys.exit(1)

            output = torch.nn.functional.softmax(outputs, dim=-1)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            acc1_meter.update(acc1.item(), target.size(0))

            pbar.set_postfix(**{'loss': loss_all / (data_iter_step + 1),
                                'lr': get_lr(optimizer),
                                'acc1': acc1_meter.avg})
            pbar.update(1)