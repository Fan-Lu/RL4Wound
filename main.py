####################################################
# Description: trainer of different algorithm
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-10-04
###################################################

import os
import random
import time
import sys

import numpy as np
import pandas as pd
from collections import deque

import matplotlib
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torch.autograd.functional import jacobian
from torch.utils.tensorboard import SummaryWriter

from algs.ode54 import Transformer, Agent_Trans
from algs.a2c import Agent_A2C
from algs.dqn import Agent_DQN
from envs.env import WoundEnv, SimpleEnv, LinearEnv, dynamics, linear_dynamic
from cfgs.config import GetParameters


def train(colab_dir, max_ep=500, args=None):
    # directories checking and creation
    alg_name = 'map524'
    args.model_dir = colab_dir + '/res_map524/models_{}/'.format(alg_name)
    args.data_dir = colab_dir + '/res_map524/data_{}/'.format(alg_name)
    args.figs_dir = colab_dir + '/res_map524/figs_{}/'.format(alg_name)

    dirs = [args.model_dir, args.data_dir, args.figs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    runs_dir = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + '_alg_' + alg_name
    runs_dir = args.model_dir + '../../runs_{}/{}'.format(alg_name, runs_dir)
    os.makedirs(runs_dir)

    # tensorboard writer to display all the data
    writer = SummaryWriter(log_dir=runs_dir)
    # whether or not to use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    # simulation of nonlinear sys
    nenv = WoundEnv(args)

    agent = Agent_A2C(nenv, args)
    mapper = Agent_Trans(nenv, args)

    for i_episode in range(max_ep):
        # reset the nonlinear dynamics to get nonlinear state at time 0
        st_nl = nenv.reset()

        # tracking the mean of mse, and reward
        loss_mean, mse1_mean, mse2_mean = 0.0, 0.0, 0.0
        d1 = time.time()
        for t in range(nenv.t_nums - 1):
            # convert to torch tensor: not that we only need the center wound information,
            # this is done by .reshape(5, args.n_cells)[:, 0]
            st_nl_pos0 = torch.from_numpy(st_nl.reshape(5, args.n_cells)[:, 0]).float().to(device).view(1, -1)
            # selection action based using DRL
            agent_state = st_nl_pos0.cpu().data.numpy().squeeze()
            # if we do not need any control, set u equals 0
            u = 0
            # pass action into nonlinear system
            st_tp1_nl, _, _, _ = nenv.step(u)
            # TODO: Mapper only save current states from non linear model
            mapper.step(st_nl, u, None, None, None)

            # state_next_non_linear, reward, done, info = env.step(u)
            st_nl = st_tp1_nl

        d2 = time.time()
        loss = mapper.learn()

        writer.add_scalar('Loss/train_mse', loss, i_episode)
        writer.add_scalar('Loss/Lamda', mapper.lam, i_episode)

        # print('\r TrainEp: {} \t RewardMean: {:4f} loss_total: {:.4f}'.format(
        #     i_episode, reward_mean / nenv.t_nums, loss), end="")
        print('TrainEp: {} \t loss_total: {:.4f} Time: {:.2f} sec'.format(
            i_episode / nenv.t_nums, loss, d2 - d1))
        if (i_episode+1) % 5 == 0:
            test(colab_dir, device, (mapper.model, agent), writer, i_episode, args)
            torch.save(mapper.model.state_dict(),
                       args.model_dir + 'checkpoint_ctr_{}_ep_{}.pth'.format(args.ctr, i_episode))
            torch.save(agent.model.state_dict(),
                       args.model_dir + 'checkpoint_ctr_{}_ep_{}.pth'.format(args.ctr, i_episode))


def test(colab_dir, device, models, writer, i_episode, args):

    nenv = WoundEnv(args)

    state_non_linear = nenv.reset()
    state_linear_chainrule_buf, state_linear_linearapx_buf, state_non_linear_buf, state_non_linear_est_buf = [], [], [], []
    cstate_linear_buf = []
    action_buf = []

    model, agent = models
    # print(model.Amat_masked.cpu().data.numpy())
    for t in range(nenv.t_nums - 1):
        state_non_linear_buf.append(state_non_linear.reshape(5, args.n_cells))
        state_non_linear_5 = state_non_linear.reshape(5, args.n_cells)[:, 0]
        state_non_linear_tensor = torch.from_numpy(state_non_linear_5).float().to(device).view(1, -1)
        # state_non_linear_tensor[:, 3] /= 3.0

        # selection action based using DRL
        u = 0
        state_next_non_linear, reward_2_abandon, done_2_abandon, info = nenv.step(u)

        state_linear, state_linear_AN, state_non_linear_aprx = model(state_non_linear_tensor)
        state_non_linear_est_buf.append(state_non_linear_aprx.cpu().data.numpy().reshape(5, -1))

        jac = jacobian(model.map524, state_non_linear_tensor, create_graph=True)

        # TODO: Code improvement needed
        n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = args.n_cells, args.spt, args.X_pump, args.beta, args.gamma1, args.gamma2, args.rho, args.mu, args.alphaTilt, args.power, args.kapa, args.Lam, args.DTilt, args.DTilt_n
        arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)
        dxdt = dynamics(state_non_linear, None, u, arrgs)
        dxdt_5 = dxdt.reshape(5, args.n_cells)[:, 0]
        Jn_dxdt = torch.matmul(jac[0, :, 0, :], torch.from_numpy(dxdt_5).float().to(device).view(-1, 1)).view(1, -1)

        react = nenv.theta_space[u]
        action_buf.append(react)

        cstate_linear_buf.append(state_linear.cpu().data.numpy())
        state_linear_chainrule_buf.append(Jn_dxdt.cpu().data.numpy())
        state_linear_linearapx = state_linear_AN
        state_linear_linearapx_buf.append(state_linear_linearapx.cpu().data.numpy())

        state_non_linear = state_next_non_linear

    cstate_linear_buf = np.array(cstate_linear_buf).squeeze()
    state_linear_chainrule_buf = np.array(state_linear_chainrule_buf).squeeze()
    state_linear_linearapx_buf = np.array(state_linear_linearapx_buf).squeeze()
    state_non_linear_buf = np.array(state_non_linear_buf).squeeze()
    state_non_linear_est_buf = np.array(state_non_linear_est_buf).squeeze()
    action_buf = np.array(action_buf).squeeze()
    trange = (nenv.t_span / 3.0)[:-1]

    im_scal = 0.8
    leg_pos = (1, 0.5)
    fig = plt.figure(figsize=(8, 12), num=4)
    plt.tight_layout()

    ax = fig.add_subplot(411)
    ax.plot(trange, action_buf, label='Actualtion')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(412)
    ax.plot(trange, cstate_linear_buf[:, 0], color='r', linestyle='--', label=r'$H-trans$')
    ax.plot(trange, cstate_linear_buf[:, 1], color='g', linestyle='--', label=r'$I-trans$')
    ax.plot(trange, cstate_linear_buf[:, 2], color='b', linestyle='--', label=r'$P-trans$')
    ax.plot(trange, cstate_linear_buf[:, 3], color='y', linestyle='--', label=r'$M-trans$')
    # ax.set_xlabel('time, sec')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(413)
    ax.plot(trange, state_linear_chainrule_buf[:, 0], color='r', linestyle='--', label=r'$\dot{H}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 1], color='g', linestyle='--', label=r'$\dot{I}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 2], color='b', linestyle='--', label=r'$\dot{P}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 3], color='y', linestyle='--', label=r'$\dot{M}-chain$')

    ax.plot(trange, state_linear_linearapx_buf[:, 0], color='r', linestyle='-', label=r'$\dot{H}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 1], color='g', linestyle='-', label=r'$\dot{I}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 2], color='b', linestyle='-', label=r'$\dot{P}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 3], color='y', linestyle='-', label=r'$\dot{M}-laprx$')

    # ax.set_xlabel('time, sec')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(414)
    ax.plot(trange, state_non_linear_buf[:, 0, 0], color='r', linestyle='--', label='Debris-ref')
    ax.plot(trange, state_non_linear_buf[:, 1, 0], color='g', linestyle='--', label='M1-ref')
    ax.plot(trange, state_non_linear_buf[:, 2, 0], color='b', linestyle='--', label='M2-ref')
    ax.plot(trange, state_non_linear_buf[:, 3, 0] / 3.0, color='y', linestyle='--', label='Temp-ref')
    ax.plot(trange, state_non_linear_buf[:, 4, 0], color='c', linestyle='--', label='New-ref')

    ax.plot(trange, state_non_linear_est_buf[:, 0], color='r', linestyle='-', label='Debris-est')
    ax.plot(trange, state_non_linear_est_buf[:, 1], color='g', linestyle='-', label='M1-est')
    ax.plot(trange, state_non_linear_est_buf[:, 2], color='b', linestyle='-', label='M2-est')
    ax.plot(trange, state_non_linear_est_buf[:, 3] / 3.0, color='y', linestyle='-', label='Temp-est')
    ax.plot(trange, state_non_linear_est_buf[:, 4], color='c', linestyle='-', label='New-est')

    ax.set_xlabel('time, day')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    # plt.show()
    writer.add_figure('Wound/Test_ctr_{}'.format(args.ctr), fig, i_episode)
    plt.close()


if __name__ == "__main__":
    args = GetParameters()
    args.gpu = False
    args.ctr = False
    colab_dir = "../../../ExpDataDARPA/"
    train(colab_dir, max_ep=10000, args=args)






