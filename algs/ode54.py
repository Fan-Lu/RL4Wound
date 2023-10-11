####################################################
# Description: Non-linear to linear dynamics
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-10-03
####################################################

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.functional import jacobian


from utils.memories import TransCache
from envs.env import dynamics


class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim):
        '''
        Find mapping between non-linear and linear models
        '''
        super(Transformer, self).__init__()
        torch.manual_seed(0)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc11 = nn.Linear(self.in_dim, 128)
        self.fc12 = nn.Linear(128, 256)
        self.fc13 = nn.Linear(256, 128)
        self.fc14 = nn.Linear(128, self.out_dim)

        self.fc21 = nn.Linear(self.out_dim, 128)
        self.fc22 = nn.Linear(128, 256)
        self.fc23 = nn.Linear(256, 128)
        self.fc24 = nn.Linear(128, self.in_dim)

        # self.Amat = nn.Linear(out_dim, out_dim, bias=False)
        kh, ki, kp = 0.5, 0.3, 0.1
        AInit = np.array([[kh, 0, 0, 0],
                          [kh, ki, 0, 0],
                          [0, ki, kp, 0],
                          [0, 0, kp, 0]])
        AMast = np.array([[-1, 0, 0, 0],
                          [1, -1, 0, 0],
                          [0, 1, -1, 0],
                          [0, 0, 1, 0]])

        self.outMap = torch.from_numpy(np.array([1, 1, 1, 3, 1])).float().view(1, -1)
        self.AMask = torch.from_numpy(AMast).float()
        self.Amat = nn.Parameter(torch.from_numpy(AInit).float(), requires_grad=True)
        self.Amat_masked = torch.multiply(F.relu(self.Amat), self.AMask)
        self.Amat.requires_grad = True

    def map524(self, x):
        x2z = F.leaky_relu(self.fc11(x))
        x2z = F.leaky_relu(self.fc12(x2z))
        x2z = F.leaky_relu(self.fc13(x2z))
        x2z = F.softmax(self.fc14(x2z), dim=1)
        return x2z

    def map425(self, x2z):
        z2x = F.leaky_relu(self.fc21(x2z))
        z2x = F.leaky_relu(self.fc22(z2x))
        z2x = F.leaky_relu(self.fc23(z2x))
        z2x = F.softmax(self.fc24(z2x), dim=1)

        z2x = torch.multiply(z2x, self.outMap)

        return z2x

    def matmult(self, x2z):
        # self.Amat_masked = torch.multiply(F.relu(self.Amat), self.AMask)
        # Az = torch.matmul(self.Amat_masked, x2z.T)
        self.Amat_masked = torch.multiply(F.relu(self.Amat), self.AMask)
        Az = torch.matmul(self.Amat_masked, x2z.T).T

        return Az

    def forward(self, x):
        x2z = self.map524(x)
        z2x = self.map425(x2z)
        AN = self.matmult(x2z)

        return x2z, AN, z2x


class Agent_Trans(object):

    def __init__(self, env, args):
        '''
        Advantage Actor Critic
        '''
        super(Agent_Trans, self).__init__()
        self.env = env
        self.args = args

        if not self.args.cloose_loop:
            self.state_size = env.state_size
        else:
            # TODO: To determined by the transformer
            self.state_size = args.decoder_size
        if not self.args.spt:
            if self.args.cont:
                self.action_size = env.action_space.shape[0]
            else:
                self.action_size = env.action_space.n
        else:
            self.action_size = env.action_space.shape[0]
        self.action_dim = 1
        self.batch_size = self.args.batch_size
        self.LR = self.args.LR
        self.TAU = self.args.TAU
        self.GAMMA = self.args.GAMMA

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

        self.model = Transformer(in_dim=5, out_dim=4).to(self.device)
        # TODO: Integrated into transformer
        self.model.AMask = self.model.AMask.to(self.device)
        self.model.outMap = self.model.outMap.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.LR)
        self.mse_loss = nn.MSELoss()

        self.gamma = 1.0 + 5e-4
        self.lam = 0.9
        self.min_lam = 0.01

        # Replay memory
        self.memory = TransCache(batch_size=self.env.t_nums - 1, device=self.device)

    def step(self, state, action, reward, next_state, done):
        '''

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''

        # save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        '''
        Update value parameters using given batch of experiences tuples

        :param experiences:
        :return:
        '''
        states, actions, rewards, next_states, dones = self.memory.sample()

        batch_loss = []
        for t in range(self.env.t_nums - 1):
            # current state obtained from non-linear dynamics
            st_nl = states[t, :].view(1, -1)
            u = actions[t, :].cpu().data.numpy().max()
            # only key the center position
            st_nl_p0 = st_nl.view(5, self.args.n_cells)[:, 0].view(1, -1)

            st_l, st_l_AN, st_nl_apprx = self.model(st_nl_p0)
            jac = jacobian(self.model.map524, st_nl_p0, create_graph=True)

            n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = self.args.n_cells, self.args.spt, self.args.X_pump, self.args.beta, self.args.gamma1, self.args.gamma2, self.args.rho, self.args.mu, self.args.alphaTilt, self.args.power, self.args.kapa, self.args.Lam, self.args.DTilt, self.args.DTilt_n
            arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)
            dxdt = dynamics(st_nl.cpu().data.numpy().reshape(-1), None, u, arrgs)

            # Get ODE at wound center
            dxdt_5 = torch.from_numpy(dxdt.reshape(5, self.args.n_cells)[:, 0]).float().to(self.device).view(-1, 1)
            Jn_dxdt = torch.matmul(jac[0, :, 0, :], dxdt_5).view(1, -1)

            # m1 macrophage
            m1 = st_nl_p0.cpu().data.numpy()[:, 1][0]
            B = torch.from_numpy(np.array([[0,  0, 0, 0, 0],
                                           [0, -m1, 0, 0, 0],
                                           [0,  m1, 0, 0, 0],
                                           [0,  0, 0, 0, 0],
                                           [0,  0, 0, 0, 0]])).float().to(self.device)
            u_tensor = torch.from_numpy(np.array([u, u, u, u, u])).float().to(self.device).view(-1, 1)
            Jn_bu = torch.matmul(jac[0, :, 0, :], torch.matmul(B, u_tensor)).view(1, -1)

            mse1 = self.mse_loss(Jn_dxdt, st_l_AN + Jn_bu)
            mse2 = self.mse_loss(st_nl_p0, st_nl_apprx)
            mse3 = self.mse_loss(st_l[:, :3], st_nl_p0[:, :3])
            loss = mse1 + self.lam * (mse2 + 0.9 * mse3)
            batch_loss.append(loss)

        self.lam = max(self.lam / self.gamma, self.min_lam)

        batch_loss = torch.vstack(batch_loss).view(-1, 1).mean()
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        # Clean memeory buffer for the next trajectory
        self.memory.reset()

        return batch_loss.cpu().data.numpy().max()