####################################################
# Description: Main Function of RL4Wound
# Version: V0.1.1
# Author: Fan Lu @ UCSC
# Data: 2023-12-29
# Architecture of Directories
'''
-- codePy
    -- algs                 # Algorithms
    -- cfgs                 # configuration files
    -- envs                 # environments
    -- res                  # experimental results
        -- data             # data saved
        -- figs             # figures
        -- models           # trained models
'''
###################################################

import os
import time

from algs.alphaHeal import AlphaHeal
from cfgs.config_device import DeviceParameters

from algs.deepmapper import DeepMapper

# pathlib: Object-oriented filesystem paths
from pathlib import Path

if __name__ == '__main__':

    wound_num = input('Please select wound number, for example: 1'
                      '\n Wound #: ')
    wound_num = int(wound_num.replace(' ', ''))
    print('Wound number is set to {} !!!'.format(wound_num))

    desktop_dir = os.path.expanduser("~/Desktop/")
    deviceArgs = DeviceParameters(wound_num, desktop_dir)

    alg_dir = desktop_dir + 'Close_Loop_Actuation/'
    alg_out = alg_dir + 'Output/'
    alg_res = alg_dir + 'data_save/'

    runs_device = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + 'wound_{}'.format(deviceArgs.wound_num)
    runs_device = alg_dir + '/runs/exvivo/wound_{}_{}/{}'.format(deviceArgs.wound_num, deviceArgs.curr_datetime, runs_device)
    deviceArgs.runs_device = runs_device

    dirs = [alg_out, alg_res, runs_device]

    for tmp in dirs:
        if not os.path.exists(tmp):
            os.makedirs(tmp)
    deviceArgs.close_loop = True
    wound = AlphaHeal(deviceArgs=deviceArgs)
    wound.control()
