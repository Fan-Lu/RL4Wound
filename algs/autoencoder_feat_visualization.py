

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from keras.utils import img_to_array

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# define the NN architecture
class Autoencoder(nn.Module):
    def __init__(self, h_dim=4):
        super(Autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # conv layer (depth from 8 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        # conv layer (depth from 4 --> 1), 3x3 kernels
        # self.conv3 = nn.Conv2d(4, 1, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        self.enc_fc1 = nn.Linear(4 * 32 * 32, 128)
        self.enc_fc2 = nn.Linear(128, h_dim)
        # self.enc_fc3 = nn.Linear(64, h_dim)

        self.dec_fc1 = nn.Linear(h_dim, 128)
        self.dec_fc2 = nn.Linear(128, 4 * 32 * 32)

        self.A_fc1 = nn.Linear(h_dim, 64)
        self.A_fc2 = nn.Linear(64, 64)
        self.A_fc3 = nn.Linear(64, 3)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        # self.t_conv1 = nn.ConvTranspose2d(1, 4, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_bn1 = nn.BatchNorm2d(16)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        self.t_b2 = nn.BatchNorm2d(3)

        # sampling time
        # self.sample_time = 0.1001
        # use 0.1 for invivo 23
        # self.sample_time = 0.05
        self.sample_time = 0.1

        self.Kh = 0.5
        self.Ki = 0.3
        self.Kp = 0.1

        self.Amat_masked = torch.zeros((3, 4, 4))
        self.Amat_masked[0][0][0] = -1.0
        self.Amat_masked[0][1][0] =  1.0
        self.Amat_masked[1][1][1] = -1.0
        self.Amat_masked[1][2][1] =  1.0
        self.Amat_masked[2][2][2] = -1.0
        self.Amat_masked[2][3][2] =  1.0

        self.z_pre = torch.from_numpy(np.array([1., 0., 0., 0.])).view(1, -1).float()
        self.softmax = nn.Softmax(dim=1)

        self.mina = torch.from_numpy(np.array([0.5, 0.1, 0.001]))
        self.maxa = torch.from_numpy(np.array([0.9, 0.5, 0.1]))

    def encoder(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # compressed representation
        encoder_maxpool_x1 = x
        # add second hidden layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # compressed representation
        encoder_maxpool_x2 = x
        # add third hidden layer

        x = x.view(1, -1)
        x = F.relu(self.enc_fc1(x))
        # x = F.softmax(self.enc_fc2(x), dim=1)
        x = F.sigmoid(self.enc_fc2(x))
        # x = F.relu(self.enc_fc2(x))

        return x, encoder_maxpool_x1, encoder_maxpool_x2

    def decoder(self, z):
        ## decode ##

        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = z.view(1, 4, 32, 32)
        # add transpose conv layers, with relu activation function
        z = F.relu(self.t_bn1(self.t_conv1(z)))
        # add transpose conv layers, with relu activation function
        z = F.sigmoid(self.t_conv2(z))

        return z

    def shift(self, z, time_dif):
        # z_dff = (z - self.z_pre) / self.sample_time * (time_dif / 7200.0)
        # self.z_pre = z.detach()

        ks = F.relu(self.A_fc1(z))
        ks = F.relu(self.A_fc2(ks))
        ks = F.sigmoid(self.A_fc3(ks))
        ks = torch.clip(ks, self.mina, self.maxa).float()

        Amat = torch.tensordot(ks.reshape(1, 1, -1), self.Amat_masked, dims=[[2], [0]]).squeeze()
        # default sample time is 7200 seconds
        Az = z + self.sample_time * (time_dif / 7200.0) * torch.matmul(Amat, z.T).T

        self.Kh, self.Ki, self.Kp = ks.cpu().data.squeeze().numpy()

        return Az.clip(0, 1)

    def forward(self, x, time_dif):
        z, ep1, ep2 = self.encoder(x)
        x_hat = self.decoder(z)
        z_next = self.shift(z, time_dif)
        x_next_hat = self.decoder(z_next)

        return z, z_next, x_hat, x_next_hat, ep1, ep2


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x


def gray_scale_feature(feature_map):
    feature_map = feature_map.squeeze(0)
    # gray_scale = torch.sum(feature_map,0)
    # gray_scale = gray_scale / feature_map.shape[0]
    return feature_map

def save_feature_map(model, im_dir, dir_exp, dir_cam, date_dir):
    # Constants
    avg_dv = np.array([108.16076384, 61.49104917, 55.44175686])
    # patch cropping size

    im_dir_tmp1 = 'E:/Data/Porcine_Exp_Davis_Processed_B/FeatureMap1/{}/{}/{}/'.format(dir_exp, dir_cam, date_dir)
    im_dir_tmp2 = 'E:/Data/Porcine_Exp_Davis_Processed_B/FeatureMap2/{}/{}/{}/'.format(dir_exp, dir_cam, date_dir)

    if not os.path.exists(im_dir_tmp1):
        os.makedirs(im_dir_tmp1)

    if not os.path.exists(im_dir_tmp2):
        os.makedirs(im_dir_tmp2)

    # device = torch.device("cuda:0" if torch.cuda.is_available() and False else "cpu")
    # model = Autoencoder().to(device)
    # model.load_state_dict(torch.load('../models/deepmapper_ep_final_in_vivo_23.pth'))

    if os.path.exists(im_dir_tmp2 + str(im_dir).split('\\')[-1]):
        return

    device_image = img_to_array(Image.open(im_dir))
    img_avg = device_image.mean(axis=(0, 1))
    device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)

    device_image = np.expand_dims(device_image.T, axis=0)
    device_image = torch.from_numpy(device_image / 255.0).float().to(device)
    with torch.no_grad():
        _, _, _, _, ep1, ep2 = model(device_image, 7200)
    gray_ep1 = gray_scale_feature(ep1)
    gray_ep2 = gray_scale_feature(ep2)

    # First Layer Features
    fig = plt.figure(figsize=(16, 16))
    for i in range(1, 17):
        ax = fig.add_subplot(4, 4, i)
        imgplottmp = plt.imshow(gray_ep1[i - 1])
        ax.axis("off")
    plt.savefig(im_dir_tmp1 + str(im_dir).split('\\')[-1], bbox_inches='tight')
    plt.close()

    # First Layer Features
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        imgplottmp = plt.imshow(gray_ep2[i - 1])
        ax.axis("off")
    plt.savefig(im_dir_tmp2 + str(im_dir).split('\\')[-1], bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    from tqdm import tqdm
    main_dir = 'E:/data/Porcine_Exp_Davis_Processed_B/Downsample/'
    device = torch.device("cuda:0" if torch.cuda.is_available() and False else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('../models/deepmapper_ep_final_in_vivo_23.pth'))

    for dir_exp in tqdm(os.listdir(main_dir), desc='Exp: '):
        if dir_exp.startswith('.'):
            continue
        dir_exp_cam = main_dir + dir_exp + '/'
        # print('Training with {} images \t ep: {}/{}'.format(dir_exp_cam, ep, mapper.num_epochs))
        for dir_cam in os.listdir(dir_exp_cam):
            if dir_cam.startswith('.'):
                continue
            dir_exp_cam_date = dir_exp_cam + dir_cam + '/'

            for date_dir in os.listdir(dir_exp_cam_date):
                dir_exp_cam_date_date = dir_exp_cam_date + date_dir + '/'
                root_images = Path(dir_exp_cam_date_date)
                image_paths = list(root_images.glob("*.jpg"))
                # avg_loss = one_trajectory(look_ahead_cnt, image_paths, mapper)
                image_paths.sort()
                for idx in range(len(image_paths)):
                    # if not os.path.exists(image_paths[idx]):
                    save_feature_map(model, image_paths[idx], dir_exp, dir_cam, date_dir)

