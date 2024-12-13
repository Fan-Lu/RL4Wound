####################################################
# Description: Deep Mapper
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-12-22
####################################################

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from keras.utils import img_to_array
from skimage import color as skcolor
from skimage import filters as skfilters

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F

# import densenet as dn
from algs.autoencoder import Autoencoder, ConvAutoencoder
from cfgs.config_healnet import HealNetParameters

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

import time

# pathlib: Object-oriented filesystem paths
from pathlib import Path

from envs.env import simple
from scipy.integrate import odeint

import sys

from utils.ims import merge_zstack

# desktop = os.path.join('./', 'Desktop')
# # PurePath subclass that can make system calls
# root_images = Path(f"../../../WoundDataDARPA/Porcine_Exp_Davis/Wound_6_merged_downsample/")
# image_paths = list(root_images.glob("*.jpg"))

from utils.tools import *

# Constants
avg_dv = np.array([108.16076384,  61.49104917,  55.44175686])
# patch cropping size
crop_size = 1024
max_noise_level = 10000


class DeepMapper(object):

    def __init__(self, deviceArgs, writer):
        super(DeepMapper, self).__init__()

        self.deviceArgs = deviceArgs

        self.args = HealNetParameters()
        self.wound_num = deviceArgs.wound_num

        self.args.model_dir = deviceArgs.desktop_dir + 'Close_Loop_Actuation/data_save/exp_{}/deepmapper/models_wound_{}/'.format(deviceArgs.expNum, self.wound_num)
        self.args.data_dir = deviceArgs.desktop_dir + 'Close_Loop_Actuation/data_save/exp_{}/deepmapper/data_wound_{}/'.format(deviceArgs.expNum, self.wound_num)
        self.args.figs_dir = deviceArgs.desktop_dir + 'Close_Loop_Actuation/data_save/exp_{}/deepmapper/figs_wound_{}/'.format(deviceArgs.expNum, self.wound_num)

        self.imdata_dir = self.args.data_dir + 'dsmgIMs/'

        dirs = [self.args.data_dir, self.args.figs_dir, self.args.model_dir, self.imdata_dir]
        for dirtmp in dirs:
            if not os.path.exists(dirtmp):
                os.makedirs(dirtmp)

        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.gpu else "cpu")
        # create models
        self.model = Autoencoder().to(self.device)
        # if deviceArgs.
        if not deviceArgs.isTrain:
            print('Finish loading DNN models...')
            sys.stdout.flush()
            self.model.load_state_dict(torch.load(deviceArgs.desktop_dir + 'Close_Loop_Actuation/models/deepmapper_ep_final.pth'))

        self.MSELoss = nn.MSELoss()
        self.CRELoss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # we assume that whent the wound is created, it has prob h one
        self.init_prob = torch.from_numpy(np.array([1., 0., 0., 0.])).view(1, -1).float().to(self.device)
        self.fina_prob = torch.from_numpy(np.array([0., 0.05, 0.15, 0.8])).view(1, -1).float().to(self.device)

        try:
            last_prbs = pd.read_csv(self.deviceArgs.deepmapper_prob_file_name)
            self.A_Prob = torch.from_numpy(last_prbs.iloc[-1][4:8].values.astype(np.float32))
        except:
            self.A_Prob = None

    def ds_merge(self, im_dir, gen_dirs=None):
        '''
        dwonsampling merging
        @im_dir: image directory where device image stored
        @return:
        '''

        im_name = im_dir.__str__().split('/')[-1]
        if gen_dirs is None:
            im_name = self.imdata_dir + im_name + '.png'
        else:
            im_name = gen_dirs + im_name + '.png'

        if not os.path.exists(im_name):
            final_image, final_coeffs_R, final_coeffs_G, final_coeffs_B = merge_zstack(im_dir)
            final_image = final_image.resize((128, 128))
            final_image.save(im_name)
        else:
            final_image = Image.open(im_name)
        return final_image

    def process_im(self, image):
        # convert a PIL Image instance to a NumPy array: shape: w x h x c
        device_image = img_to_array(Image.open(image))
        img_avg = device_image.mean(axis=(0, 1))
        # device_image = np.clip(device_image +
        #                        np.expand_dims(avg_dv - img_avg, axis=0) +
        #                        np.random.uniform(0, 1, device_image.shape), 0, 255).astype(int)
        device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)

        return device_image

    def test(self, ep, image_dir, progressor=None):
        im_gens = []
        im_orgs = []
        image_dir.sort()
        prob_buf = np.array([1., 0., 0., 0.])
        time_process = str2sec(str(image_dir[0])[-20:-5]) - 7200.0
        allks = []

        for idx in range(len(image_dir)):
            # xrange.append((xrange[-1] + 1) * 2)
            curr_device_image = self.process_im(image_dir[idx])
            curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
            curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(self.device)

            time_dif = str2sec(str(image_dir[idx])[-20:-5]) - time_process
            time_process = str2sec(str(image_dir[idx])[-20:-5])

            prob, A_prob, x_hat, x_next_hat = self.model(curr_image_data, time_dif)
            allks.append([self.model.Kh, self.model.Ki, self.model.Kp])

            prob_buf = np.vstack((prob_buf, A_prob.cpu().data.numpy().squeeze()))

            x_hat_np = x_hat.data.numpy().squeeze().T
            x_hat_np = x_hat_np * 255
            im_hat = Image.fromarray(x_hat_np.astype(np.uint8))
            im_org = Image.fromarray((curr_image_data.data.numpy().squeeze().T * 255).astype(np.uint8))

            im_gens.append(im_hat)
            im_orgs.append(im_org)
        allks = np.array(allks).reshape(-1, 3)

        numIms = len(im_orgs)
        numRows = numIms // 7
        if numIms % 7 > 0:
            numRows += 1

        dst1 = Image.new('RGB', (128 * 7, 128 * numRows))
        dst2 = Image.new('RGB', (128 * 7, 128 * numRows))
        for j in range(numRows):
            for i in range(7):
                if (i + j * 7) < numIms:
                    dst1.paste(im_orgs[i + j * 7], (i * 128, 128 * j))
                    dst2.paste(im_gens[i + j * 7], (i * 128, 128 * j))

        # if progressor is not None:
        xrange = [hr / 12.0 for hr in range(len(prob_buf))]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(211)
        ax.step(xrange[1:], prob_buf[1:, 0], color='r', label='H')
        ax.step(xrange[1:], prob_buf[1:, 1], color='g', label='I')
        ax.step(xrange[1:], prob_buf[1:, 2], color='b', label='P')
        ax.step(xrange[1:], prob_buf[1:, 3], color='y', label='M')
        leg_pos = (1, 0.5)
        ax.legend(loc='center left', bbox_to_anchor=leg_pos)
        ax.set_xlabel('Time (day)')

        ax = fig.add_subplot(212)
        ax.step(xrange[1:], allks[:, 0], color='r', label='kh')
        ax.step(xrange[1:], allks[:, 1], color='g', label='ki')
        ax.step(xrange[1:], allks[:, 2], color='b', label='kp')
        ax.legend(loc='center left', bbox_to_anchor=leg_pos)
        ax.set_xlabel('Time (day)')
        plt.tight_layout()

        # probdf = pd.DataFrame(prob_buf)
        # probdf.to_csv('./data_save/exp_{}/probs_wound_{}_ep_{}.csv'.format(
        #     self.deviceArgs.expNum, self.deviceArgs.wound_num, self.deviceArgs.wound_num, ep))
        #
        # allkdf = pd.DataFrame(allks)
        # allkdf.to_csv('./data_save/exp_{}/allks_wound_{}_ep_{}.csv'.format(
        #     self.deviceArgs.expNum, self.deviceArgs.wound_num, self.deviceArgs.wound_num, ep))

        if self.writer is not None:
            self.writer.add_figure('wsd_stage/prob', fig, ep)
            self.writer.add_image('wsd_stage/im_orgs', img_to_array(dst1) / 255.0, ep, dataformats='HWC')
            self.writer.add_image('wsd_stage/im_gens', img_to_array(dst2) / 255.0, ep, dataformats='HWC')
        else:
            plt.savefig('./data_save/exp_{}/stages_wound_{}_ep_{}.png'.format(
            self.deviceArgs.expNum, self.deviceArgs.wound_num, self.deviceArgs.wound_num, ep))

        plt.close()
        dst1.close()
        dst2.close()

    def test_psu(self, ep, image_dir):
        im_gens = []
        im_orgs = []
        image_dir.sort()
        prob_buf = np.array([1., 0., 0., 0.])
        time_process = str2sec(str(image_dir[0])[-20:-5]) - 7200.0
        allks = []

        for idx in range(84):

            if idx < len(image_dir):

                # xrange.append((xrange[-1] + 1) * 2)
                curr_device_image = self.process_im(image_dir[idx])
                curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
                curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(self.device)

                time_dif = str2sec(str(image_dir[idx])[-20:-5]) - time_process
                time_process = str2sec(str(image_dir[idx])[-20:-5])

                prob, A_prob, x_hat, x_next_hat = self.model(curr_image_data, time_dif)
                allks.append([self.model.Kh, self.model.Ki, self.model.Kp])

                prob_buf = np.vstack((prob_buf, A_prob.cpu().data.numpy().squeeze()))

                x_hat_np = x_hat.data.numpy().squeeze().T
                x_hat_np = x_hat_np * 255
                im_hat = Image.fromarray(x_hat_np.astype(np.uint8))
                im_org = Image.fromarray((curr_image_data.data.numpy().squeeze().T * 255).astype(np.uint8))

                im_gens.append(im_hat)
                im_orgs.append(im_org)
            else:
                A_prob = self.model.shift(A_prob, 7200.0)
                allks.append([self.model.Kh, self.model.Ki, self.model.Kp])
                prob_buf = np.vstack((prob_buf, A_prob.cpu().data.numpy().squeeze()))

        allks = np.array(allks).reshape(-1, 3)

        probs_smoothed = smoother(prob_buf[:, 1])

        numIms = len(im_orgs)
        numRows = numIms // 7
        if numIms % 7 > 0:
            numRows += 1

        dst1 = Image.new('RGB', (128 * 7, 128 * numRows))
        dst2 = Image.new('RGB', (128 * 7, 128 * numRows))
        for j in range(numRows):
            for i in range(7):
                if (i + j * 7) < numIms:
                    dst1.paste(im_orgs[i + j * 7], (i * 128, 128 * j))
                    dst2.paste(im_gens[i + j * 7], (i * 128, 128 * j))

        # if progressor is not None:
        xrange = [hr / 12.0 for hr in range(len(prob_buf))]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(211)
        ax.plot(xrange[1:], prob_buf[1:, 0], color='r', label='H')
        # ax.plot(xrange[1:], prob_buf[1:, 1], color='g', label='I')
        ax.plot(xrange[1:], prob_buf[1:, 2], color='b', label='P')
        ax.plot(xrange[1:], prob_buf[1:, 3], color='y', label='M')

        ax.plot(xrange[1:], probs_smoothed[1:], color='g', linestyle='dashed')

        leg_pos = (1, 0.5)
        ax.legend(loc='center left', bbox_to_anchor=leg_pos)
        ax.set_xlabel('Time (day)')

        ax = fig.add_subplot(212)
        ax.step(xrange[1:], allks[:, 0], color='r', label='kh')
        ax.step(xrange[1:], allks[:, 1], color='g', label='ki')
        ax.step(xrange[1:], allks[:, 2], color='b', label='kp')
        ax.legend(loc='center left', bbox_to_anchor=leg_pos)
        ax.set_xlabel('Time (day)')
        plt.tight_layout()

        # probdf = pd.DataFrame(prob_buf)
        # probdf.to_csv('./data_save/exp_{}/probs_wound_{}_ep_{}.csv'.format(
        #     self.deviceArgs.expNum, self.deviceArgs.wound_num, self.deviceArgs.wound_num, ep))
        #
        # allkdf = pd.DataFrame(allks)
        # allkdf.to_csv('./data_save/exp_{}/allks_wound_{}_ep_{}.csv'.format(
        #     self.deviceArgs.expNum, self.deviceArgs.wound_num, self.deviceArgs.wound_num, ep))

        plt.savefig('./data_save/exp_{}/stages_wound_{}_ep_{}.png'.format(
        self.deviceArgs.expNum, self.deviceArgs.wound_num, self.deviceArgs.wound_num, ep))

        plt.close()
        dst1.close()
        dst2.close()

    def ws_est(self, im_dir):
        '''
        Wound stage estimation
        @im_dir: (dir) the latest wound image directory where device image stored
        @return:
        '''

        # merging and downsampling
        print('Merging generated images of {}...'.format(im_dir.split('/')[-1]))
        sys.stdout.flush()
        wound_image = self.ds_merge(im_dir)
        print('Finish image generation.')
        sys.stdout.flush()
        device_image = img_to_array(wound_image)
        img_avg = device_image.mean(axis=(0, 1))
        device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)
        device_image = np.expand_dims(device_image.T, axis=0)
        device_image = torch.from_numpy(device_image / 255.0).float().to(self.device)
        probs, A_prob, x_hat, x_next_hat = self.model(device_image)
        A_prob = A_prob.cpu().data.numpy().squeeze()

        return A_prob

    def ws_est_gen(self, im_dir, time_dif):
        '''
        Wound stage estimation
        @im_dir: (dir) the latest wound image directory where device image stored
        @return:
        '''

        # merging and downsampling
        print('Merging generated images of {}...'.format(im_dir.split('/')[-1]))
        sys.stdout.flush()
        wound_image = self.ds_merge(im_dir)
        print('Finish image generation.')
        sys.stdout.flush()
        device_image = img_to_array(wound_image)
        img_avg = device_image.mean(axis=(0, 1))
        device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)
        device_image = np.expand_dims(device_image.T, axis=0)
        device_image = torch.from_numpy(device_image / 255.0).float().to(self.device)

        # if len(os.listdir(im_dir + '/..')) <= 30 or self.A_Prob is None:
        if self.A_Prob is None:

            device_image = img_to_array(wound_image)
            img_avg = device_image.mean(axis=(0, 1))
            device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)
            device_image = np.expand_dims(device_image.T, axis=0)
            device_image = torch.from_numpy(device_image / 255.0).float().to(self.device)
            probs, A_prob, x_hat, x_next_hat = self.model(device_image, time_dif)
            self.A_Prob = A_prob
            A_prob = A_prob.cpu().data.numpy().squeeze()
        else:
            self.A_Prob = self.model.shift(self.A_Prob, time_dif)
            A_prob = self.A_Prob.cpu().data.numpy().squeeze()
            x_hat = device_image

        return A_prob, device_image.cpu().data.numpy().squeeze().T, x_hat.cpu().data.numpy().squeeze().T