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

import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from algs.alphaHeal import AlphaHeal
from cfgs.config_device import DeviceParameters


if __name__ == '__main__':

    # invivo = input('Is this invivo test or ex-invivo? 1 means invivo, 0 means ex-invivo'
    #                   '\n invivo #: ')
    # invivo = bool(int(invivo.replace(' ', '')))
    # print('Experiment is set to {} !!!'.format(invivo))

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
    runs_device = alg_dir + 'runs/invivo_exp_{}/wound_{}_{}/{}'.format(deviceArgs.expNum, deviceArgs.wound_num, deviceArgs.curr_datetime, runs_device)
    deviceArgs.runs_device = runs_device

    treatment = 3

    if treatment == 0:
        print('No Treatment will be applied')
    elif treatment == 1:
        print('Treatment is set to EF delivery!!!')
        max_EF_current = input('Please set maximum current (default is 10muA), for example: 100'
                            '\n Max EF Current: ')
        max_EF_current = float(max_EF_current)
        deviceArgs.maxEFCurrent = max_EF_current
        print('Max current is set to {} muA !!!'.format(max_EF_current))
    elif treatment == 2:
        print('Treatment is set to Flx delivery')
        max_Flx_current = input('Please set maximum current (default is 10muA), for example: 100'
                            '\n Max Flx Current: ')
        max_Flx_current = float(max_Flx_current)
        deviceArgs.maxFlxCurrent = max_Flx_current
        print('Max current is set to {} muA !!!'.format(max_Flx_current))

        max_dosage = input('Please set maximum dosage to deliver in a day (default is 0.025mg/day), for example: 0.025'
                           '\n MaxDosage: ')
        if len(max_dosage) != 0:
            max_dosage = float(max_dosage)
            deviceArgs.maxDosage = max_dosage
        print('Max dosage is set to {} mg !!!'.format(deviceArgs.maxDosage))
    elif treatment == 3:
        print('Treatment is set to Flx/EF delivery')

        max_EF_current = input('Please set maximum current (default is 10muA), for example: 100'
                            '\n Max EF Current: ')
        max_EF_current = float(max_EF_current)
        deviceArgs.maxEFCurrent = max_EF_current
        print('Max current for EF is set to {} muA !!!'.format(deviceArgs.maxEFCurrent))

        max_Flx_current = input('Please set maximum current (default is 10muA), for example: 100'
                            '\n Max Flx Current: ')
        max_Flx_current = float(max_Flx_current)
        deviceArgs.maxFlxCurrent = max_Flx_current
        print('Max current for Flx is set to {} muA !!!'.format(deviceArgs.maxFlxCurrent))

        max_dosage = input('Please set maximum dosage to deliver in a day (default is 0.45mg), for example: 0.45'
                           '\n MaxDosage: ')
        if len(max_dosage) != 0:
            max_dosage = float(max_dosage)
            deviceArgs.maxDosage = max_dosage
        print('Max dosage is set to {} mg !!!'.format(deviceArgs.maxDosage))
    else:
        assert False, 'Please Specify the correct treatment type! Either type in 1 or 2!'

    deviceArgs.treatment = treatment
    # deviceArgs.invivo = invivo

    dirs = [alg_out, alg_res, runs_device]

    for tmp in dirs:
        if not os.path.exists(tmp):
            os.makedirs(tmp)

    deviceArgs.invivo = False
    wound = AlphaHeal(deviceArgs=deviceArgs)
    if treatment == 0:
        wound.nocotrl()
    elif treatment == 1 or treatment == 2:
        wound.control()
    elif treatment == 3:
        # EF first, then Flx
        wound.comControlOpt12()
    else:
        assert False, print("Please Select the right treatment")
