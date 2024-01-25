####################################################
# Description: Config file for device
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-12-12
####################################################

import time
from datetime import datetime
import argparse
import os


def DeviceParameters(wound_no, desktop_dir):

    parser = argparse.ArgumentParser(description='Device Setting')

    # device related parameters
    parser.add_argument('--wound_num', default=wound_no, type=int,
                        help='wound number')

    # file directories
    curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    healnet_prob_file_name = desktop_dir + 'HealNet-Inference/prob_table.csv'
    target_current_file_name = desktop_dir + 'Close_Loop_Actuation/Wound_{}.csv'.format(wound_no)
    # The Close_Loop_Actuation folder is for Prahbat
    prabhat_cv_file_name = desktop_dir + 'Close_Loop_Actuation/Output/Wound_{}.csv'.format(wound_no)
    fl_curent_file_name = desktop_dir + 'Close_Loop_Actuation/data_save/Wound_{}_{}.csv'.format(wound_no, curr_datetime)
    fl_err_file_name = desktop_dir + 'Close_Loop_Actuation/data_save/error_{}_{}.csv'.format(wound_no, curr_datetime)
    device_im_dir = desktop_dir + 'Porcine_Exp_Davis/'

    runs_device = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + 'wound_{}'.format(wound_no)
    runs_device = desktop_dir + 'Close_Loop_Actuation' + '/runs_device/wound_{}_{}/{}'.format(wound_no, curr_datetime, runs_device)

    parser.add_argument('--curr_datetime', default=curr_datetime, type=str, help='current date and time')
    parser.add_argument('--desktop_dir', default=desktop_dir, type=str, help='desktop_dir')
    parser.add_argument('--healnet_prob_file_name', default=healnet_prob_file_name, type=str, help='healnet_prob_file_name')
    parser.add_argument('--target_current_file_name', default=target_current_file_name, type=str, help='target_current_file_name')
    parser.add_argument('--prabhat_cv_file_name', default=prabhat_cv_file_name, type=str, help='prabhat_cv_file_name')
    parser.add_argument('--fl_curent_file_name', default=fl_curent_file_name, type=str, help='fl_curent_file_name')
    parser.add_argument('--fl_err_file_name', default=fl_err_file_name, type=str, help='fl_err_file_name')
    parser.add_argument('--runs_device', default=runs_device, type=str, help='tensorboard data directory')
    parser.add_argument('--device_im_dir', default=device_im_dir, type=str, help='directory to device images')

    # open-loop control strategy
    parser.add_argument('--open_trigger', default=0.75, type=float,
                        help='when proliferation reaches 0.75, trigger open loop control')

    parser.add_argument('--response_time_threshold', default=60.0, type=float, help='response_time_threshold = 60')
    parser.add_argument('--low_target', default=5.0, type=float,
                        help='set before hours/2')
    parser.add_argument('--high_target', default=10.0, type=float,
                        help='set after hours/2')
    parser.add_argument('--heal_target', default=7.0, type=float,
                        help='set when healnet is invoked during the first hours/2')
    parser.add_argument('--n_chs', default=8, type=int,
                        help='total number of channel')

    parser.add_argument('--maxCurrent', default=10.0, type=float,
                        help='maximum value of current in muA')

    parser.add_argument('--close_loop', default=False, type=bool, help='whether or not use closed loop control')
    # def control frequency
    parser.add_argument('--delta_t', default=5.0, type=float, help='control every 5 seconds')

    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)

    # args = parser.parse_args()
    # a dummy argument to fool ipython
    args, unknown = parser.parse_known_args()

    return args