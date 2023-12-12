import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nets.model import Model
from utils.WashingtonDataset import WashingtonDataset
from utils.train_one_epoch import train_one_epoch
from utils.evaluate import evaluate
from utils.util import NativeScalerWithGradNormCount as NativeScaler

# Use a function to set up the device for better readability
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args_parser():
    parser = argparse.ArgumentParser('CNN_TransNet', add_help=False)
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for DataLoader')
    parser.add_argument('--data_type', default='colorized_depth')
    parser.add_argument('--optimizer_type', default='SGD', choices=['adam', 'SGD'], help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset_dir', default='./data', help='Path for dataset')
    parser.add_argument('--output_dir', default='./output', help='Path for saving output')
    parser.add_argument('--log_dir', default='./output', help='Path for saving logs')
    parser.add_argument('--train_file', default='data/splits/trial_1_train.txt', help='Training file path')
    parser.add_argument('--test_file', default='data/splits/trial_1_test.txt', help='Test file path')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for DataLoader efficiency')
    parser.set_defaults(pin_memory=True)
    return parser


def main(args):
    device = get_device()

    # Read dataset files
    with open(args.train_file, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.test_file, encoding='utf-8') as f:
        test_lines = f.readlines()

    train_dataset = WashingtonDataset(train_lines, args.dataset_dir)
    val_dataset = WashingtonDataset(test_lines, args.dataset_dir)

    # Use the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True)

    model = Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Improved optimizer selection
    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
                 'SGD': torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)}[args.optimizer_type]

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)
    loss_scaler = NativeScaler()
    epoch_step_train = len(train_lines) // args.batch_size
    epoch_step_test = len(test_lines) // args.batch_size

    for epoch in range(args.epochs):
        print(f'Starting Epoch {epoch + 1}/{args.epochs}')

        # Combined training and evaluation
        model.train()
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, loss_scaler, epoch_step_train)

        model.eval()
        test_stats = evaluate(val_loader, model, device, epoch, epoch_step_test)
        print(f"Accuracy on test set: {test_stats['acc1']:.1f}%")
        print(f"Loss on test set: {test_stats['loss']:.2f}")

        if log_writer:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        if args.output_dir:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch + 1}.pth"))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
