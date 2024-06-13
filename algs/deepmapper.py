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
            self.model.load_state_dict(torch.load(deviceArgs.desktop_dir + 'Close_Loop_Actuation/models/deepmapper_ep_final.pth'))

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.num_epochs = 1000
        self.ndays = 8

        # self.xrange = np.linspace(0, self.ndays, 12 * 10)
        self.xrange = np.linspace(0, self.ndays, 77)

        # we assume that whent the wound is created, it has prob h one
        self.init_prob = torch.from_numpy(np.array([1., 0., 0., 0.])).view(1, -1).float().to(self.device)


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
        xrange = np.linspace(0, 3, len(image_dir) + 1)
        prob_buf = np.array([1., 0., 0., 0.])
        time_process = str2sec(str(image_dir[0])[-20:-5]) - 7200.0
        err = 0
        grd = 0

        pgs = []

        if progressor is not None:
            pgs.append(progressor(prob_buf.reshape(1, -1))[0]/ 20.0)

        for idx in range(len(image_dir)):
            # xrange.append((xrange[-1] + 1) * 2)
            curr_device_image = self.process_im(image_dir[idx])
            curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
            curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(self.device)

            time_dif = str2sec(str(image_dir[idx])[-20:-5]) - time_process
            time_process = str2sec(str(image_dir[idx])[-20:-5])

            prob, A_prob, x_hat, x_next_hat = self.model(curr_image_data, time_dif)

            self.writer.add_scalar('Ks_test/k_h', self.model.Kh, idx)
            self.writer.add_scalar('Ks_test/k_i', self.model.Ki, idx)
            self.writer.add_scalar('Ks_test/k_p', self.model.Kp, idx)

            if progressor is not None:
                pgs.append(progressor(prob.data.numpy().reshape(1, -1))[0]/ 20.0)
            prob_buf = np.vstack((prob_buf, A_prob.cpu().data.numpy().squeeze()))

            x_hat_np = x_hat.data.numpy().squeeze().T
            x_hat_np = x_hat_np * 255
            im_hat = Image.fromarray(x_hat_np.astype(np.uint8))
            im_org = Image.fromarray((curr_image_data.data.numpy().squeeze().T * 255).astype(np.uint8))

            err += np.sum(np.abs(x_hat_np - (curr_image_data.data.numpy().squeeze().T * 255)))
            grd += np.sum(x_hat_np)

            im_gens.append(im_hat)
            im_orgs.append(im_org)

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

        xrangeAll = np.linspace(0, 20, 240)

        # TODO: Fix Xrange
        y_tmp = odeint(simple, np.array([1., 0., 0., 0.]), xrangeAll,
                       args=(np.array([self.model.Kh, self.model.Kh, self.model.Kh])))

        if progressor is not None:
            xrange = [hr / 12.0 for hr in range(len(prob_buf))]
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(211)
            ax.step(xrange, prob_buf[:, 0], color='r', label='H')
            ax.step(xrange, prob_buf[:, 1], color='g', label='I')
            ax.step(xrange, prob_buf[:, 2], color='b', label='P')
            ax.step(xrange, prob_buf[:, 3], color='y', label='M')
            ax.legend()

            ax = fig.add_subplot(212)
            ax.step(xrange, pgs, color='r', label='Prg')

            leg_pos = (1, 0.5)
            ax.legend(loc='center left', bbox_to_anchor=leg_pos)
            ax.set_xlabel('Time (day)')
        else:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.scatter(xrange, prob_buf[:, 0], color='r')  # , label='Hemostasis')
            ax.scatter(xrange, prob_buf[:, 1], color='g')  # , label='Inflammation')
            ax.scatter(xrange, prob_buf[:, 2], color='b')  # , label='Proliferation')
            ax.scatter(xrange, prob_buf[:, 3], color='y')  # , label='Maturation')

            ax.plot(xrangeAll, y_tmp[:, 0], color='r', label='H')
            ax.plot(xrangeAll, y_tmp[:, 1], color='g', label='I')
            ax.plot(xrangeAll, y_tmp[:, 2], color='b', label='P')
            ax.plot(xrangeAll, y_tmp[:, 3], color='y', label='M')

            leg_pos = (1, 0.5)
            ax.legend(loc='center left', bbox_to_anchor=leg_pos)
            ax.set_xlabel('Time (day)')

        plt.savefig('./data_save/exp_{}/probs_{}.pdf'.format(self.deviceArgs.expNum, self.deviceArgs.wound_num), format='pdf')

        if progressor is not None:
            df = {'H': prob_buf[:, 0],
                  'I': prob_buf[:, 1],
                  'P': prob_buf[:, 2],
                  'M': prob_buf[:, 3],
                  "progress": np.array(pgs)}
            probdf = pd.DataFrame(df)
        else:
            probdf = pd.DataFrame(prob_buf)
        probdf.to_csv('./data_save/exp_{}/probs_{}.csv'.format(self.deviceArgs.expNum, self.deviceArgs.wound_num))

        self.writer.add_figure('wsd_stage/prob', fig, ep)
        self.writer.add_image('wsd_stage/orgs', img_to_array(dst1) / 255.0, ep, dataformats='HWC')
        self.writer.add_image('wsd_stage/gens', img_to_array(dst2) / 255.0, ep, dataformats='HWC')

        plt.close()
        dst1.close()
        dst2.close()

    def train(self):
        look_ahead_cnt = 4
        # image_paths = [os.listdir(self.imdata_dir)]
        root_images = Path(self.imdata_dir)
        image_paths = list(root_images.glob("*.png"))

        for ep in range(self.num_epochs):
            avg_loss = 0.0
            cnt = 0
            for idx in tqdm(range(len(image_paths)), position=0, leave=True):

                if idx < len(image_paths) - look_ahead_cnt:
                    curr_device_image = self.process_im(image_paths[idx])
                    next_device_image = self.process_im(image_paths[idx + 1])
                    next4_device_image = self.process_im(image_paths[idx + look_ahead_cnt])

                    curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
                    curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(self.device)

                    next_image_data = np.expand_dims(next_device_image.T, axis=0)
                    next_image_data = torch.from_numpy(next_image_data / 255.0).float().to(self.device)

                    next4_image_data = np.expand_dims(next4_device_image.T, axis=0)
                    next4_image_data = torch.from_numpy(next4_image_data / 255.0).float().to(self.device)

                    prob, A_prob, x_hat, x_next_hat = self.model(curr_image_data)
                    prob_next, _, _, _ = self.model(next_image_data)
                    prob_next4, _, _, _ = self.model(next4_image_data)
                    self.optimizer.zero_grad()

                    A_prob_shift = self.model.shift(A_prob)
                    for xxx in range(look_ahead_cnt - 1):
                        A_prob_shift = self.model.shift(A_prob_shift)

                    if idx <= 0:
                        loss = 0.2 * self.criterion(x_hat, curr_image_data) + \
                               0.2 * self.criterion(x_next_hat, next_image_data) + \
                               0.4 * self.criterion(prob, self.init_prob) + \
                               0.05 * self.criterion(A_prob, prob_next.detach()) + \
                               0.05 * self.criterion(prob_next, A_prob.detach()) + \
                               0.1 * self.criterion(A_prob_shift, prob_next4)
                    else:
                        loss = 0.2 * self.criterion(x_hat, curr_image_data) + \
                               0.2 * self.criterion(x_next_hat, next_image_data) + \
                               0.15 * self.criterion(A_prob, prob_next.detach()) + \
                               0.15 * self.criterion(prob_next, A_prob.detach()) + \
                               0.3 * self.criterion(A_prob_shift, prob_next4)

                    loss.backward()
                    self.optimizer.step()

                    avg_loss += loss.cpu().item()
                    cnt += 1

                else:
                    break

            if ep % 1 == 0:
                print('Train Epoch [{}/{}] Loss:{:.4f}'.format(ep + 1, self.num_epochs, avg_loss / cnt))

                if ep % 10 == 0:
                    self.test(ep, image_paths)

                torch.save(self.model.state_dict(), self.args.model_dir + 'checkpoint_ep_{}.pth'.format(ep))
                self.writer.add_scalar('Loss/train_mse', avg_loss / cnt, ep)
                self.writer.add_scalar('Ks/k_h', F.sigmoid(self.model.Kh).data.cpu().numpy()[0], ep)
                self.writer.add_scalar('Ks/k_i', F.sigmoid(self.model.Ki).data.cpu().numpy()[0], ep)
                self.writer.add_scalar('Ks/k_p', F.sigmoid(self.model.Kp).data.cpu().numpy()[0], ep)

    def ws_est(self, im_dir):
        '''
        Wound stage estimation
        @im_dir: (dir) the latest wound image directory where device image stored
        @return:
        '''

        # merging and downsampling
        print('Merging generated images of {}...'.format(im_dir.split('/')[-1]))
        wound_image = self.ds_merge(im_dir)
        print('Finish image generation.')
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
        wound_image = self.ds_merge(im_dir)
        print('Finish image generation.')
        device_image = img_to_array(wound_image)
        img_avg = device_image.mean(axis=(0, 1))
        device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)
        device_image = np.expand_dims(device_image.T, axis=0)
        device_image = torch.from_numpy(device_image / 255.0).float().to(self.device)
        probs, A_prob, x_hat, x_next_hat = self.model(device_image, time_dif)
        A_prob = A_prob.cpu().data.numpy().squeeze()

        return A_prob, device_image.cpu().data.numpy().squeeze().T, x_hat.cpu().data.numpy().squeeze().T