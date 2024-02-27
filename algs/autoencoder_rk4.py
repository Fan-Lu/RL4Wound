

from curses import KEY_B2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # self.dec_fc3 = nn.Linear(128, 4 * 32 * 32)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        # self.t_conv1 = nn.ConvTranspose2d(1, 4, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_b1 = nn.BatchNorm2d(16)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        self.t_b2 = nn.BatchNorm2d(3)

        # sampling time
        # self.sample_time = 0.1001
        self.sample_time = 0.85

        KhMask = np.array([[-1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

        KiMask = np.array([[0, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 0]])

        KpMask = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 1, 0]])

        self.KhMask = torch.from_numpy(KhMask).float()
        self.KiMask = torch.from_numpy(KiMask).float()
        self.KpMask = torch.from_numpy(KpMask).float()

        # self.Amat = nn.Parameter(torch.from_numpy(AInit).float(), requires_grad=True)

        self.Kh = nn.Parameter(torch.from_numpy(np.array([0.5])).float(), requires_grad=True)
        self.Ki = nn.Parameter(torch.from_numpy(np.array([0.3])).float(), requires_grad=True)
        self.Kp = nn.Parameter(torch.from_numpy(np.array([0.1])).float(), requires_grad=True)

        # self.Amat_masked = torch.multiply(F.relu(self.Amat), self.AMask)
        self.Amat_masked = torch.multiply(self.Kh, self.KhMask) + \
                           torch.multiply(self.Ki, self.KiMask) + \
                           torch.multiply(self.Kp, self.KpMask)

        self.softmax = nn.Softmax(dim=1)

    def encoder(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # compressed representation
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        # add third hidden layer

        x = x.view(1, -1)
        x = F.relu(self.enc_fc1(x))
        x = F.softmax(self.enc_fc2(x), dim=1)
        # x = F.relu(self.enc_fc2(x))

        return x

    def decoder(self, z):
        ## decode ##

        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = z.view(1, 4, 32, 32)
        # add transpose conv layers, with relu activation function
        z = F.relu(self.t_conv1(z))
        # add transpose conv layers, with relu activation function
        z = F.sigmoid(self.t_conv2(z))

        return z

    def shift(self, z):
        # self.Amat_masked = torch.clip(torch.multiply(F.relu(self.Amat), self.AMask), 0.1, 1.0)

        self.Amat_masked = torch.multiply(F.sigmoid(self.Kh), self.KhMask) + \
                           torch.multiply(F.sigmoid(self.Ki), self.KiMask) + \
                           torch.multiply(F.sigmoid(self.Kp), self.KpMask)

        k1 = self.sample_time*torch.matmul(self.Amat_masked, z.T) # was transposed again in Fan's original code, may need to transpose to use in def of k2
        k2 = self.sample_time*torch.matmul(self.Amat_masked, 0.5*k1 + z.T)
        k3 = self.sample_time*torch.matmul(self.Amat_masked, 0.5*k2 + z.T)
        k4 = self.sample_time*torch.matmul(self.Amat_masked, k3 + z.T) 

        z_next = z + (self.sample_time/6)*(k1 + 2*k2 + 2*k3 + k4).T # this transpose call may or may not be needed.

        return z_next

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        z_next = self.shift(z)
        x_next_hat = self.decoder(z_next)

        return z, z_next, x_hat, x_next_hat


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
