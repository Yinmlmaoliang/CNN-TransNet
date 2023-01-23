import os
import torch
import argparse
import random
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from net.model import Model
from utils.WashingtonDataset import WashingtonDataset
from utils.train_one_epoch import train_one_epoch
from utils.evaluate import evaluate
from utils.util import NativeScalerWithGradNormCount as NativeScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args_parser():
    parser = argparse.ArgumentParser('CNN_TransNet', add_help=False)
    parser.add_argument('--img_size', default=224, type=int, help='image size')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--num_works', default=8, type=int)
    parser.add_argument('--data_type', default='colorized_depth')
    parser.add_argument('--optimizer_type', default='SGD', help='type:adam,SGD')
    parser.add_argument('--weight_decay', type=int, default=0.0001,
                        help='Weight decay (default:0.05')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--output_dir', default='./output_dir_pretrained',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_pretrained',
                        help='path where to save tensorboard log')
    parser.add_argument('--train_file',
                        default='trial_1_train.txt',
                        help='training file path')
    parser.add_argument('--test_file',
                        default='trail_1_test.txt',
                        help='test file path')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin cpu memory in Dataloader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):
    with open(args.train_file, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.test_file, encoding='utf-8') as f:
        test_lines = f.readlines()

    train_dataset = WashingtonDataset(train_lines)
    val_dataset = WashingtonDataset(test_lines)

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                            num_workers=args.num_works, pin_memory=args.pin_mem, drop_last=True)
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                          num_workers=args.num_works, pin_memory=args.pin_mem, drop_last=True)

    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                          weight_decay=args.weight_decay),
                 'SGD': torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                        nesterov=True)}[args.optimizer_type]

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)
    loss_scaler = NativeScaler()

    for epoch in range(args.epochs):
        if epoch % 1 == 0:
            print('Evaluating...')
            model.eval()
            test_stats = evaluate(val_data, model, device)
            print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        model.train()
        print('Training...')

        epoch_step = num_train // args.batch_size

        train_one_epoch(model, criterion, train_data, optimizer, device, epoch, loss_scaler, epoch_step)
        if args.output_dir is None:
            print('Saving checkpoints...')
            torch.save(model.state_dict(), os.path.join(args.output_dir, "epoch%03d.pth" % (epoch + 1)))

        if epoch == args.epochs - 1:
            print('Evaluating...')
            model.eval()
            test_stats = evaluate(val_data, model, device)
            print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%")

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)