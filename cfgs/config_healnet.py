####################################################
# Description: Config file for HealNet
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-12-12
####################################################


import argparse

def HealNetParameters():

    parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')

    parser.add_argument('--t_days', default=60, type=int, help='Total number of days')
    parser.add_argument('--t_nums', default=601, type=int, help='Dived t_days into t_nums')

    parser.add_argument('--gpu', default=True, type=bool, help='Dived t_days into t_nums')

    parser.add_argument('--action_size', default=10, type=int, help='Dived t_days into t_nums')

    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--layers', default=100, type=int,
                        help='total number of layers (default: 100)')
    parser.add_argument('--growth', default=12, type=int,
                        help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0, type=float,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--reduce', default=0.5, type=float,
                        help='compression rate in transition stage (default: 0.5)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                        help='To not use bottleneck block')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                        help='name of experiment')
    parser.add_argument('--tensorboard',
                        help='Log progress to TensorBoard', action='store_true')
    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)

    # args = parser.parse_args()
    # a dummy argument to fool ipython
    args, unknown = parser.parse_known_args()

    return args