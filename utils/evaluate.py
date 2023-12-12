import torch
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm


def evaluate(data_loader, model, device, epoch, epoch_step):
    criterion = torch.nn.CrossEntropyLoss()
    acc1_meter = AverageMeter()
    loss_meter = AverageMeter()
    # switch to evaluation mode
    model.eval()
    with tqdm(total=epoch_step, desc=f'Test: ', postfix=dict, mininterval=0.3) as pbar:
        for data_iter_step, batch in enumerate(data_loader):
            rgb = batch[0]
            depth = batch[1]
            target = batch[-1]
            
            rgb = rgb.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.no_grad():
                output = model(rgb, depth)

                loss = criterion(output, target)

                output = torch.nn.functional.softmax(output, dim=-1)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                acc1_meter.update(acc1.item(), target.size(0))
                loss_meter.update(loss.item(), target.size(0))

            pbar.set_postfix(**{'loss': loss_meter.avg,
                                'acc1': acc1_meter.avg})
            pbar.update(1)
    return {'loss': loss_meter.avg, 'acc1': acc1_meter.avg}