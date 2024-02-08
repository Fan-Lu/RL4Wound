####################################################
# Description: HealNet
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-11-29
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

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from keras.utils import img_to_array
from skimage import color as skcolor
from skimage import filters as skfilters

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn

# import densenet as dn
from algs.autoencoder import Autoencoder, ConvAutoencoder
from cfgs.config_healnet import HealNetParameters

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

import time

# pathlib: Object-oriented filesystem paths
from pathlib import Path


from algs.deepmapper import DeepMapper
from cfgs.config_device import DeviceParameters


if __name__ == "__main__":

    np.random.seed(0)
    torch.random.manual_seed(0)
    # wound_num = input('Please select wound number, for example: 1'
    #                   '\n Wound #: ')
    wound_num = 6
    print('Wound number is set to {} !!!'.format(wound_num))

    desktop_dir = os.path.expanduser("~/Desktop/")
    deviceArgs = DeviceParameters(wound_num, desktop_dir)

    alg_dir = desktop_dir + 'Close_Loop_Actuation/'
    alg_out = alg_dir + 'Output/'
    alg_res = alg_dir + 'data_save/'

    runs_device = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + 'wound_{}'.format(deviceArgs.wound_num)
    runs_device = alg_dir + '/runs/exvivo/wound_{}_{}/{}'.format(deviceArgs.wound_num, deviceArgs.curr_datetime, runs_device)
    deviceArgs.runs_device = runs_device
    writer = SummaryWriter(log_dir=deviceArgs.runs_device)
    model = DeepMapper(deviceArgs, writer)
    model.model.sample_time = 0.2
    model.train()
