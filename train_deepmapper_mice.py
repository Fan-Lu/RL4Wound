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


from envs.env import SimpleEnv, simple
from scipy.integrate import odeint

desktop = os.path.join('./', 'Desktop')
# PurePath subclass that can make system calls
root_images = Path(f"../../../WoundDataDARPA/Porcine_Exp_Davis/Wound_6_merged_downsample/")
image_paths = list(root_images.glob("*.jpg"))

# Constants
avg_dv = np.array([108.16076384,  61.49104917,  55.44175686])
# patch cropping size
crop_size = 1024
max_noise_level = 10000


def preprocess(image):
    # convert a PIL Image instance to a NumPy array: shape: w x h x c
    device_image = img_to_array(Image.open(image).resize((128, 128)))
    img_avg = device_image.mean(axis=(0, 1))
    device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)

    return device_image


def create_imdirs_from_csv(age, cnt, side, train=True):
    imdir = 'D:/WoundDataDARPA/MouseData/'

    im_paths = []
    if train:
        df = pd.read_csv(imdir + 'all_train_imgs.csv')
    else:
        df = pd.read_csv(imdir + 'all_test_imgs.csv')
    df_tmp = df.loc[(df.Age == age) & (df.WNum == cnt) & (df.Side == side)]

    for i in range(len(df_tmp)):
        if train:
            dir_tmp = imdir + 'train/' + '{}/'.format(df_tmp.Day.iloc[i]) + df_tmp.ImNa.iloc[i]
        else:
            dir_tmp = imdir + 'val/' + '{}/'.format(df_tmp.Day.iloc[i]) + df_tmp.ImNa.iloc[i]
        im_paths.append(dir_tmp)
    if len(im_paths) > 4:
        return im_paths
    else:
        return None


def test(ep, device, model, writer, image_paths):

    im_gens = []
    im_orgs = []

    prob_buf = np.array([1., 0., 0., 0.])
    for idx in range(len(image_paths)):
        curr_device_image = preprocess(image_paths[idx])
        curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
        curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(device)

        prob, A_prob, x_hat, x_next_hat = model(curr_image_data)
        prob_buf = np.vstack((prob_buf, A_prob.cpu().data.numpy().squeeze()))

        x_hat_np = x_hat.data.numpy().squeeze().T
        x_hat_np = x_hat_np * 255
        im_hat = Image.fromarray(x_hat_np.astype(np.uint8))
        im_org = Image.fromarray((curr_image_data.data.numpy().squeeze().T * 255).astype(np.uint8))

        im_gens.append(im_hat)
        im_orgs.append(im_org)

    dst1 = Image.new('RGB', (128 * 7, 128 * 7))
    dst2 = Image.new('RGB', (128 * 7, 128 * 7))
    for j in range(7):
        for i in range(7):
            if (i + j * 7) < len(im_gens):
                dst1.paste(im_orgs[i + j * 7], (i * 128, 128 * j))
                dst2.paste(im_gens[i + j * 7], (i * 128, 128 * j))

    xrange = np.linspace(0, len(im_orgs), len(prob_buf[:, 0]))

    y_tmp = odeint(simple, np.array([1., 0., 0., 0.]), xrange, args=(np.array([F.sigmoid(model.Kh).data.cpu().numpy()[0],
                                                                      F.sigmoid(model.Ki).data.cpu().numpy()[0],
                                                                      F.sigmoid(model.Kp).data.cpu().numpy()[0]]),))
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax.scatter(xrange, prob_buf[:, 0], color='r') #, label='Hemostasis')
    ax.scatter(xrange, prob_buf[:, 1], color='g') #, label='Inflammation')
    ax.scatter(xrange, prob_buf[:, 2], color='b') #, label='Proliferation')
    ax.scatter(xrange, prob_buf[:, 3], color='y') #, label='Maturation')

    ax.plot(xrange, y_tmp[:, 0], color='r', label='H')
    ax.plot(xrange, y_tmp[:, 1], color='g', label='I')
    ax.plot(xrange, y_tmp[:, 2], color='b', label='P')
    ax.plot(xrange, y_tmp[:, 3], color='y', label='M')

    leg_pos = (1, 0.5)
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax.set_xlabel('Time (day)')

    writer.add_figure('wsd_stage/prob', fig, ep)
    writer.add_image('wsd_stage/orgs', img_to_array(dst1) / 255.0, ep, dataformats='HWC')
    writer.add_image('wsd_stage/gens', img_to_array(dst2) / 255.0, ep, dataformats='HWC')
    plt.close()
    dst1.close()
    dst2.close()


def wound_predict(model, device, image_paths):
    im_gens = []
    prob_buf = np.array([1., 0., 0., 0.])
    for idx in range(len(image_paths)):
        curr_device_image = preprocess(image_paths[idx])
        curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
        curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(device)

        prob, A_prob, x_hat, x_next_hat = model(curr_image_data)
        prob_buf = np.vstack((prob_buf, A_prob.cpu().data.numpy().squeeze()))

        x_hat_np = x_hat.data.numpy().squeeze().T
        x_hat_np = x_hat_np * 255
        im_hat = Image.fromarray(x_hat_np.astype(np.uint8))
        im_gens.append(im_hat)

        genname = "/".join(image_paths[idx].split('/')[:-1]) + "/gen_" + image_paths[idx].split('/')[-1]
        im_hat.save(genname)

    return prob_buf, im_gens


def report(ep, device):

    image_paths_Y8 = create_imdirs_from_csv("Y8", 4, 'R', False)
    image_paths_A8 = create_imdirs_from_csv("A8", 1, 'L', False)

    model_A8 = Autoencoder().to(device)
    model_Y8 = Autoencoder().to(device)
    model_A8.load_state_dict(torch.load('D:/res/models/deepmapper/checkpoint_mouse_age_A8_ep_{}.pth'.format(ep)))
    model_Y8.load_state_dict(torch.load('D:/res/models/deepmapper/checkpoint_mouse_age_Y8_ep_{}.pth'.format(ep)))

    prob_buf_Y8, im_gens_Y8 = wound_predict(model_Y8, device, image_paths_Y8)
    prob_buf_A8, im_gens_A8 = wound_predict(model_A8, device, image_paths_A8)

    mTdays = min(len(prob_buf_Y8[:, 0]), len(prob_buf_A8[:, 0]))
    xrange = np.linspace(0, mTdays, mTdays)

    y_tmp_A8 = odeint(simple,
                      np.array([1., 0., 0., 0.]),
                      xrange,
                      args=(np.array([F.sigmoid(model_A8.Kh).data.cpu().numpy()[0],
                                      F.sigmoid(model_A8.Ki).data.cpu().numpy()[0],
                                      F.sigmoid(model_A8.Kp).data.cpu().numpy()[0]]),))
    y_tmp_Y8 = odeint(simple,
                      np.array([1., 0., 0., 0.]),
                      xrange,
                      args=(np.array([F.sigmoid(model_Y8.Kh).data.cpu().numpy()[0],
                                      F.sigmoid(model_Y8.Ki).data.cpu().numpy()[0],
                                      F.sigmoid(model_Y8.Kp).data.cpu().numpy()[0]]),))

    prob_name = ['H', 'I', 'P', 'M']
    colors = ['r', 'g', 'b', 'y']

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    for pi in range(4):
        ax.plot(xrange, y_tmp_A8[:, pi], color=colors[pi], linestyle='--', label=prob_name[pi])
        ax.plot(xrange, y_tmp_Y8[:, pi], color=colors[pi], linestyle='-', label=prob_name[pi])
    lines = ax.get_lines()
    legend_prob = plt.legend([lines[i] for i in [1, 3, 5, 7]], prob_name, loc=(0.8, 0.5))
    legend_age = plt.legend([lines[i] for i in [6, 7]], ['Adult', 'Young'], loc=(0.8, 0.3))
    ax.add_artist(legend_prob)
    ax.add_artist(legend_age)
    ax.set_xlabel('Time (day)')
    plt.tight_layout()
    plt.savefig('theoretical_solution.png')
    plt.close()


    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(221)
    ax.plot(xrange, prob_buf_Y8[:len(xrange), 0], color=colors[0], linestyle='-', label='Young Hemostasis')
    ax.plot(xrange, prob_buf_A8[:len(xrange), 0], color=colors[0], linestyle='--', label='Adult Hemostasis')
    ax.scatter(xrange, prob_buf_Y8[:len(xrange), 0], color=colors[0], marker='*')
    ax.scatter(xrange, prob_buf_A8[:len(xrange), 0], color=colors[0], marker='x')
    ax.legend()
    ax.set_xlabel('Time (day)')

    ax = fig.add_subplot(222)
    ax.plot(xrange, prob_buf_Y8[:len(xrange), 1], color=colors[1], linestyle='-', label='Young Inflammation')
    ax.plot(xrange, prob_buf_A8[:len(xrange), 1], color=colors[1], linestyle='--', label='Adult Inflammation')
    ax.scatter(xrange, prob_buf_Y8[:len(xrange), 1], color=colors[1], marker='*')
    ax.scatter(xrange, prob_buf_A8[:len(xrange), 1], color=colors[1], marker='x')
    ax.legend()
    ax.set_xlabel('Time (day)')

    ax = fig.add_subplot(223)
    ax.plot(xrange, prob_buf_Y8[:len(xrange), 2], color=colors[2], linestyle='-', label='Young Proliferation')
    ax.plot(xrange, prob_buf_A8[:len(xrange), 2], color=colors[2], linestyle='--', label='Adult Proliferation')
    ax.scatter(xrange, prob_buf_Y8[:len(xrange), 2], color=colors[2], marker='*')
    ax.scatter(xrange, prob_buf_A8[:len(xrange), 2], color=colors[2], marker='x')
    ax.legend()
    ax.set_xlabel('Time (day)')

    ax = fig.add_subplot(224)
    ax.plot(xrange, prob_buf_Y8[:len(xrange), 3], color=colors[3], linestyle='-', label='Young Maturation')
    ax.plot(xrange, prob_buf_A8[:len(xrange), 3], color=colors[3], linestyle='--', label='Adult Maturation')
    ax.scatter(xrange, prob_buf_Y8[:len(xrange), 3], color=colors[3], marker='*')
    ax.scatter(xrange, prob_buf_A8[:len(xrange), 3], color=colors[3], marker='x')
    ax.legend()
    ax.set_xlabel('Time (day)')
    plt.tight_layout()
    plt.savefig('deepmapper_healnet.png')
    plt.close()


def train(age='A8'):
    args = HealNetParameters()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    args.model_dir = './res/models/deepmapper/'
    args.data_dir = './res/data/deepmapper/'
    args.figs_dir = './res/figs/deepmapper/'

    dirs = [args.data_dir, args.figs_dir, args.model_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    runs_dir = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + '_alg_' + 'deepmapper_{}'.format(age)
    runs_dir = './res/runs/runs_deepmapper/{}'.format(runs_dir)
    os.makedirs(runs_dir)

    writer = SummaryWriter(log_dir=runs_dir)

    # create model
    model = Autoencoder().to(device)

    # define loss function (criterion) and pptimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    num_epochs = 50000

    # we assume that whent the wound is created, it has prob h one
    init_prob = torch.from_numpy(np.array([1., 0., 0., 0.])).view(-1, 4).float().to(device)

    for ep in tqdm(range(num_epochs), position=0, leave=True):
        avg_loss = 0.0
        cnt = 0

        imdir = 'E:/data/MouseData/'
        df = pd.read_csv(imdir + 'all_train_imgs.csv')

        wnums = set(df.loc[(df.Age == age)].WNum.values)
        sides = ['L', 'R']
        look_ahead_cnt = 3

        for wn in wnums:
            for side in sides:
                image_paths = create_imdirs_from_csv(age, wn, side)
                if image_paths is None or len(image_paths) < look_ahead_cnt:
                    continue
                optimizer.zero_grad()
                for idx in range(len(image_paths)):
                    if idx < len(image_paths) - look_ahead_cnt:
                        curr_device_image = preprocess(image_paths[idx])
                        next_device_image = preprocess(image_paths[idx + 1])
                        next_n_device_image = preprocess(image_paths[idx + look_ahead_cnt])

                        curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
                        curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(device)

                        next_image_data = np.expand_dims(next_device_image.T, axis=0)
                        next_image_data = torch.from_numpy(next_image_data / 255.0).float().to(device)

                        next_n_image_data = np.expand_dims(next_n_device_image.T, axis=0)
                        next_n_image_data = torch.from_numpy(next_n_image_data / 255.0).float().to(device)

                        prob, A_prob, x_hat, x_next_hat = model(curr_image_data)
                        prob_next_n, _, x_hat_n, _ = model(next_n_image_data)
                        with torch.no_grad():
                            prob_next, _, _, _ = model(next_image_data)

                        A_prob_shift = prob
                        for xxx in range(look_ahead_cnt):
                            A_prob_shift = model.shift(A_prob_shift)

                        loss = criterion(x_hat, curr_image_data) + \
                               criterion(x_next_hat, next_image_data) + \
                               criterion(x_hat_n, next_n_image_data) + \
                               criterion(A_prob, prob_next.detach()) + \
                               criterion(prob_next_n, A_prob_shift.detach())

                        # TODO: 1. Add constraints on zero time
                        #       2. Add constraints on final time
                        # loss += criterion(prob_init, mapper.init_prob)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        avg_loss += loss.cpu().item()
                        cnt += 1


        if (ep + 1) % 100 == 0:
            print('Train Epoch [{}/{}] Loss:{:.4f}'.format(ep + 1, num_epochs, avg_loss / cnt))
            if age == "A8":
                image_paths = create_imdirs_from_csv(age, 1, 'L', False)
            else:
                image_paths = create_imdirs_from_csv(age, 4, 'R', False)
            test(ep, device, model, writer, image_paths)
            torch.save(model.state_dict(), args.model_dir + 'checkpoint_mouse_age_{}_ep_{}.pth'.format(age, ep))
            writer.add_scalar('Loss/train_mse', avg_loss / cnt, ep)
            writer.add_scalar('Ks/k_h', F.sigmoid(model.Kh).data.cpu().numpy()[0], ep)
            writer.add_scalar('Ks/k_i', F.sigmoid(model.Ki).data.cpu().numpy()[0], ep)
            writer.add_scalar('Ks/k_p', F.sigmoid(model.Kp).data.cpu().numpy()[0], ep)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() and False else "cpu")
    np.random.seed(0)
    torch.random.manual_seed(0)
    # age = input('Age of Mounse: A8 or Y8 \n')
    age = 'A8'
    train(age)


    # dir_models_A8 = 'D:/res/models/deepmapper/checkpoint_mouse_age_A8_ep_3192.pth'
    # dir_models_Y8 = 'D:/res/models/deepmapper/checkpoint_mouse_age_Y8_ep_3192.pth'
    # report(6999, device)