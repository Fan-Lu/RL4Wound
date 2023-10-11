####################################################
# Description: Environments
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-12
####################################################

import numpy as np
import pandas as pd

import random

from control import lqr
import gym
from scipy.integrate import odeint
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy import linalg
import numba as nb
import sys
from numba import njit, prange

from numba import jit

def actuator(t, t1, dt):
    y = 0
    if t > t1 and t < t1 + dt:
        y = 1
    if t > t1 and t < t1 + 1:
        y = t - t1
    if t > t1 + dt - 1 and t < t1 + dt:
        y = -t + t1 + dt
    return y


# def ion_concentration(loc_x):
#     '''
#
#     :param loc_x:
#     :return:
#     '''
#     amplitude = 1.0
#     n = 1.25
#     # TODO: change n_cells from 100 to 10
#     # x_z = loc_x / 10.0
#     x_z = loc_x
#     return amplitude * np.exp(-((x_z ** 2) / n))

# @jit(nopython=True)

@njit(parallel=False)
def dynamicsAcc(z, t, action, arrgs):
    '''
    Dynamics of wound healing
    :param y:   list with length 5
                y[0]: a
                y[1]: Macrophages M1
                y[2]: Macrophages M1
                y[3]: Temporary tissue
                y[4]: New Tissue
    :param X_pump: position of ion pump that creates EF, attracting macrophages
    :param n_cells:
    :return: dydt
    '''

    n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = arrgs


    dadt_l = np.empty(n_cells)
    dm1dt_l = np.empty(n_cells)
    dm2dt_l = np.empty(n_cells)
    dcdt_l = np.empty(n_cells)
    dndt_l = np.empty(n_cells)

    amplitude = 1.0
    n = 1.25

    # if not spt:
    #     theta = action
    # # both temporal and spatio control
    # else:
    #     theta, pos = action[0], int(action[1])
    #     args.X_pump = pos
    theta = action

    for i in prange(n_cells):

        x_z = (i - X_pump) / 10.0
        tetta = amplitude * np.exp(-((x_z ** 2) / n))

        at, m1t, m2t, ct, nt = (z[i],                       # debris
                                z[i + n_cells],        # M1
                                z[i + 2 * n_cells],    # M2
                                z[i + 3 * n_cells],    # temp tissue
                                z[i + 4 * n_cells])    # new tissue
        dadt = -at * m1t
        dm1dt = (beta * at
                 - at * m1t
                 - gamma1 * m1t
                 - rho * ((m1t ** power) / ((kapa ** power) + (m1t ** power)))
                 - theta * (tetta * m1t))
        dm2dt = (rho * ((m1t ** power) / ((kapa ** power) + (m1t ** power)))
                 - gamma2 * m2t
                 + theta * (tetta * m1t))
        dcdt = m2t - mu * ct
        dndt = ct * (alphaTilt * nt * (1 - nt))
        dadt_l[i], dm1dt_l[i], dm2dt_l[i], dcdt_l[i], dndt_l[i] = dadt, dm1dt, dm2dt, dcdt, dndt

    dm1dt_l[0] += DTilt * (z[1 + n_cells] - z[n_cells]) / (Lam * Lam)
    dm2dt_l[0] += DTilt * (z[1 + 2 * n_cells] - z[2 * n_cells]) / (Lam * Lam)
    dndt_l[0] += DTilt_n * (z[3 * n_cells] * (z[1 + 4 * n_cells] - z[4 * n_cells])) / (Lam ** 2)

    # second derivative
    for i in prange(1, n_cells - 1):
        dm1dt_l[i] += DTilt * (z[i - 1 + n_cells] - 2 * z[i + n_cells] + z[i + 1 + n_cells]) / (Lam ** 2)
        dm2dt_l[i] += DTilt * (z[i - 1 + 2 * n_cells] - 2 * z[i + 2 * n_cells] + z[i + 1 + 2 * n_cells]) / (Lam ** 2)

        dndt_l[i] += DTilt_n * (z[i + 3 * n_cells] * (z[i - 1 + 4 * n_cells] - 2 * z[i + 4 * n_cells] + z[i + 1 + 4 * n_cells])) / (Lam ** 2)
    n_c = n_cells - 1
    dm1dt_l[n_c] += DTilt * (z[n_c - 1 + n_cells] - z[n_c + n_cells]) / (Lam ** 2)
    dm2dt_l[n_c] += DTilt_n * (z[n_c - 1 + 2 * n_cells] - z[n_c + 2 * n_cells]) / (Lam ** 2)

    dndt_l[n_c] += DTilt_n * (z[n_c + 3 * n_cells] * (z[n_c - 1 + 4 * n_cells] - 2 * z[n_c + 4 * n_cells] + 1)) / (Lam ** 2)
    # dzdt = np.concatenate([dadt_l, dm1dt_l, dm2dt_l, dcdt_l, dndt_l], axis=0)

    alldt = [dadt_l, dm1dt_l, dm2dt_l, dcdt_l, dndt_l]
    dzdt = np.empty(n_cells * len(alldt))
    idx = 0
    for xx in alldt:
        for dxdt in xx:
            dzdt[idx] = dxdt
            idx += 1

    # dzdt = np.array([dadt_l, dm1dt_l, dm2dt_l, dcdt_l, dndt_l]).reshape(-1)
    return dzdt


def dynamics(z, t, action, arrgs):
    '''
    Dynamics of wound healing
    :param y:   list with length 5
                y[0]: a
                y[1]: Macrophages M1
                y[2]: Macrophages M1
                y[3]: Temporary tissue
                y[4]: New Tissue
    :param X_pump: position of ion pump that creates EF, attracting macrophages
    :param n_cells:
    :return: dydt
    '''

    n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = arrgs


    dadt_l = np.empty(n_cells)
    dm1dt_l = np.empty(n_cells)
    dm2dt_l = np.empty(n_cells)
    dcdt_l = np.empty(n_cells)
    dndt_l = np.empty(n_cells)

    amplitude = 1.0
    n = 1.25

    # if not spt:
    #     theta = action
    # # both temporal and spatio control
    # else:
    #     theta, pos = action[0], int(action[1])
    #     args.X_pump = pos
    theta = action

    for i in range(n_cells):

        x_z = (i - X_pump) / 10.0
        tetta = amplitude * np.exp(-((x_z ** 2) / n))

        at, m1t, m2t, ct, nt = (z[i],                       # debris
                                z[i + n_cells],        # M1
                                z[i + 2 * n_cells],    # M2
                                z[i + 3 * n_cells],    # temp tissue
                                z[i + 4 * n_cells])    # new tissue
        dadt = -at * m1t
        dm1dt = (beta * at
                 - at * m1t
                 - gamma1 * m1t
                 - rho * ((m1t ** power) / ((kapa ** power) + (m1t ** power)))
                 - theta * (tetta * m1t))
        dm2dt = (rho * ((m1t ** power) / ((kapa ** power) + (m1t ** power)))
                 - gamma2 * m2t
                 + theta * (tetta * m1t))
        dcdt = m2t - mu * ct
        dndt = ct * (alphaTilt * nt * (1 - nt))
        dadt_l[i], dm1dt_l[i], dm2dt_l[i], dcdt_l[i], dndt_l[i] = dadt, dm1dt, dm2dt, dcdt, dndt

    dm1dt_l[0] += DTilt * (z[1 + n_cells] - z[n_cells]) / (Lam * Lam)
    dm2dt_l[0] += DTilt * (z[1 + 2 * n_cells] - z[2 * n_cells]) / (Lam * Lam)
    dndt_l[0] += DTilt_n * (z[3 * n_cells] * (z[1 + 4 * n_cells] - z[4 * n_cells])) / (Lam ** 2)

    # second derivative
    for i in range(1, n_cells - 1):
        dm1dt_l[i] += DTilt * (z[i - 1 + n_cells] - 2 * z[i + n_cells] + z[i + 1 + n_cells]) / (Lam ** 2)
        dm2dt_l[i] += DTilt * (z[i - 1 + 2 * n_cells] - 2 * z[i + 2 * n_cells] + z[i + 1 + 2 * n_cells]) / (Lam ** 2)

        dndt_l[i] += DTilt_n * (z[i + 3 * n_cells] * (z[i - 1 + 4 * n_cells] - 2 * z[i + 4 * n_cells] + z[i + 1 + 4 * n_cells])) / (Lam ** 2)
    n_c = n_cells - 1
    dm1dt_l[n_c] += DTilt * (z[n_c - 1 + n_cells] - z[n_c + n_cells]) / (Lam ** 2)
    dm2dt_l[n_c] += DTilt_n * (z[n_c - 1 + 2 * n_cells] - z[n_c + 2 * n_cells]) / (Lam ** 2)

    dndt_l[n_c] += DTilt_n * (z[n_c + 3 * n_cells] * (z[n_c - 1 + 4 * n_cells] - 2 * z[n_c + 4 * n_cells] + 1)) / (Lam ** 2)
    # dzdt = np.concatenate([dadt_l, dm1dt_l, dm2dt_l, dcdt_l, dndt_l], axis=0)

    alldt = [dadt_l, dm1dt_l, dm2dt_l, dcdt_l, dndt_l]
    dzdt = np.empty(n_cells * len(alldt))
    idx = 0
    for xx in alldt:
        for dxdt in xx:
            dzdt[idx] = dxdt
            idx += 1

    # dzdt = np.array([dadt_l, dm1dt_l, dm2dt_l, dcdt_l, dndt_l]).reshape(-1)
    return dzdt


def simple(z, t, action):
    kh, ki, kp = action.clip(0.0, 1.0)
    H, I, P, M = z
    dhdt = -kh * H
    didt = kh * H - ki * I
    dpdt = ki * I - kp * P
    dmdt = kp * P

    dzdt = [dhdt, didt, dpdt, dmdt]
    return dzdt


def linear_dynamic(z, t, action):
    A, B = action
    Q = np.eye(A.shape[0])
    R = np.eye(B.shape[1])
    z = z.reshape(-1, 1)

    # # Compute the algebraic Riccati equation.
    # P = linalg.solve_sylvester(A.T, -Q, A - B @ R @ B.T)
    # if not np.all(np.diag(P) > 0) or np.any(np.real(np.linalg.eigvals(A)) > 0):
    #     # print('LQR Solution Does Not Exist: Set input to zero!!!')
    #     u = np.zeros([B.shape[1], 1])
    # else:
    #     K, S, E = lqr(A, B, Q, R)
    #     u = -K @ z
    #     # print('LQR Solution Found u: {}'.format(u.squeeze()))
    try:
        K, S, E = lqr(A, B, Q, R)
        u = -K @ z
    except:
        u = np.zeros([B.shape[1], 1])

    dzdt = A @ z + B @ u
    dzdt = list(dzdt.squeeze())
    return dzdt


class WoundEnv(object):
    '''
    ### Description

    This is

    ```
    @ARTICLE{10.3389/fams.2022.791064,
    AUTHOR={Zlobina, Ksenia and Xue, Jiahao and Gomez, Marcella},
    TITLE={Effective Spatio-Temporal Regimes for Wound Treatment by
    Way of Macrophage Polarization: A Mathematical Model},
    JOURNAL={Frontiers in Applied Mathematics and Statistics},
    YEAR={2022},
    URL={https://www.frontiersin.org/articles/10.3389/fams.2022.791064},
    }
    ```

    ### Observation Space

    ### Action Space

    There are 2 discrete deterministic actions:
    | Num | Observation | Value | Unit |
    |-----|-------------|-------|------|
    | 0   | Treatment   | 0     |  N/A |
    | 1   | Non-Treat   | 1     | N/A  |

    ### Transition Dynamics

    ### Reward

    ### Starting State

    ### Terminal State

    ### Version History
    * v0: Initial version
    '''

    def __init__(self, args):
        '''

        :param args:
        '''

        # Hyper parameters
        self.args = args
        # number of regions in the wound
        self.n_cells = args.n_cells
        # position of ion pump that creates EF, attracting macrophages
        self.X_pump = args.X_pump
        self.t_nums = args.t_nums
        self.r = args.r
        self.t_days = args.t_days

        self.state_size = 5 * self.n_cells

        # TODO: Maybe each action last for several intervals?
        # self.duaration_space = np.linspace(2, 2, 1, dtype=int)
        self.theta_space = np.linspace(0, 1, args.action_size)

        if not self.args.spt:
            if args.cont:
                self.action_space = gym.spaces.Box(low=0, high=1., dtype=np.float32)
            else:
                self.action_space = gym.spaces.Discrete(n=args.action_size, start=0)
        else:
            self.action_space = gym.spaces.Box(low=np.array([0, 0], dtype=np.float32),
                                               high=np.array([1, args.n_cells], dtype=np.float32),
                                               dtype=np.float32)
        # self.action_space = []
        # for t_idx in range(len(self.theta_space)):
        #     for d_idx in range(len(self.duaration_space)):
        #         self.action_space.append([self.theta_space[t_idx], self.duaration_space[d_idx]])
        # self.action_space = np.array(self.action_space)
        # if not args.spt:
        #     self.action_dim = 1

        self.t_span = np.linspace(0, self.t_days, self.t_nums)

        # Environment Initialization
        # initial state: 0 day wound observed
        self.state_init = np.array([[1.] * self.n_cells,
                                    [0.] * self.n_cells,
                                    [0.] * self.n_cells,
                                    [0.] * self.n_cells,
                                    [0.] * self.n_cells]).reshape(-1)
        self.state = self.state_init
        self.max_inflamation = 0
        self.stage = 0
        self.cnter = 0
        self.done = False
        # TODO: Set y0
        self.y0 = None

    def ode_solver(self, action):
        '''
        5-state ODE Model
        :param action: initial condition
        :param t_span_num: span of ode, freq
        :return X: Dim: (5, n_cells)
        '''
        tspan_tmp = [self.t_span[self.cnter], self.t_span[self.cnter + 1]]
        n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = self.args.n_cells, self.args.spt, self.args.X_pump, self.args.beta, self.args.gamma1, self.args.gamma2, self.args.rho, self.args.mu, self.args.alphaTilt, self.args.power, self.args.kapa, self.args.Lam, self.args.DTilt, self.args.DTilt_n
        arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)

        if self.args.ctr:
            y_tmp = odeint(dynamicsAcc, self.state, tspan_tmp, args=(action, arrgs,))
        else:
            y_tmp = odeint(dynamics, self.state, tspan_tmp, args=(action, arrgs,))
        self.y0 = y_tmp[1]
        X = np.array(y_tmp[1, :]).reshape(5, -1)
        # X[3, :] = X[3, :] / 3.0
        return X

    def step(self, action):
        '''

        :param action:
        :return:
                state:
                reward:
                info: wound info
        '''

        if not self.args.spt:
            if not self.args.cont:
                action = self.theta_space[action]

        # X: Dim: 5 X n_cells
        X = self.ode_solver(action)
        self.cnter += 1
        reward = -1.0
        # TODO: Constrains
        # we apply treatment when inflammation at wound center reaches maximum
        # self.max_inflamation = max(self.max_inflamation, X[1][0])
        # X[4][0] < self.max_inflamation means when inflammation starts to decrease.
        if X[4][0] >= 0.95: # and X[1][0] < self.max_inflamation:
            self.done = True
            reward = 0

        # Expend into one dim vector form
        next_state = X.reshape(-1)
        self.state = next_state
        info = X

        return self.state, reward, self.done, info

    def reset(self):
        '''
        Reset Environment to intial state
        :return:
        '''
        self.state = self.state_init
        self.done = False
        self.cnter = 0
        self.max_inflamation = 0
        return self.state

    def render(self):
        pass


class SimpleEnv(object):

    def __init__(self, args):
        super(SimpleEnv, self).__init__()

        # Hyper parameters
        self.args = args
        self.t_nums = args.t_nums_sim
        self.t_days = args.t_days

        self.state_size = 4
        self.action_size = 3
        self.action_dim = 3
        self.action_space = np.linspace(0.0, 1.0, args.action_size)
        self.t_span = np.linspace(0, self.t_days, self.t_nums)

        self.k_opt = np.array([0.5, 0.3, 0.1])
        self.X_noise = None
        self.state_init = np.array([1, 0, 0, 0])
        self.state = self.y0 = self.state_init
        self.cnter = 0

    def ode_solver(self, action=None, FT=False):
        if not FT:
            if action is None:
                action = self.k_opt
            tspan_tmp = [self.t_span[self.cnter], self.t_span[self.cnter + 1]]
            y_tmp = odeint(simple, self.y0, tspan_tmp, args=(action,))
            self.y0 = y_tmp[1]
            X = np.array(y_tmp[1, :])
        else:
            tspan_tmp = self.t_span
            y_tmp = odeint(simple, self.y0, tspan_tmp, args=(action,))
            X = np.array(y_tmp)
        return X

    def step(self, action=None):
        X = self.ode_solver(action)
        next_state = X.reshape(-1)

        # reward = -np.linalg.norm(self.state - self.X_noise[:, self.cnter])
        self.cnter += 1
        next_state_real = self.X_noise[self.cnter, :]
        self.state = next_state

        reward = np.exp(-np.sqrt(
            0.25 * (next_state_real[0] - next_state[0]) ** 2 +
            0.25 * (next_state_real[1] - next_state[1]) ** 2 +
            0.25 * (next_state_real[2] - next_state[2]) ** 2 +
            0.25 * (next_state_real[3] - next_state[3]) ** 2
        )) - 1
        info = [self.state, 1]
        return self.state, reward, self.done, info

    def reset(self):
        self.state = self.y0 = self.state_init
        self.done = False
        self.cnter = 0
        self.X_noise = self.ode_solver(self.k_opt, FT=True)
        return self.state


class LinearEnv(object):

    def __init__(self, args):
        super(LinearEnv, self).__init__()
        self.args = args
        self.t_nums = args.t_nums_sim
        self.t_days = args.t_days

        self.state_size = 4
        self.t_span = np.linspace(0, self.t_days, self.t_nums)

        self.X_noise = None
        self.state_init = np.array([1, 0, 0, 0])
        self.state = self.y0 = self.state_init
        self.cnter = 0

    def ode_solver(self, action=None, FT=False):
        if not FT:
            tspan_tmp = [self.t_span[self.cnter], self.t_span[self.cnter + 1]]
            y_tmp = odeint(linear_dynamic, self.y0, tspan_tmp, args=(action,))
            self.y0 = y_tmp[1]
            X = np.array(y_tmp[1, :])
        else:
            tspan_tmp = self.t_span
            y_tmp = odeint(linear_dynamic, self.y0, tspan_tmp, args=(action,))
            X = np.array(y_tmp)
        return X

    def step(self, action=None):
        X = self.ode_solver(action)
        next_state = X.reshape(-1)

        # reward = -np.linalg.norm(self.state - self.X_noise[:, self.cnter])
        self.cnter += 1
        self.state = next_state

        return self.state

    def reset(self):
        self.state = self.y0 = self.state_init
        self.done = False
        self.cnter = 0
        return self.state


if __name__ == '__main__':
    from cfgs.config import GetParameters

    args = GetParameters()
    env = WoundEnv(args)
    state = env.reset()
    # action = (0.5, 0.3, 0.1)
    # t_span = np.linspace(0, args.t_days, args.t_nums)
    # n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = args.n_cells, args.spt, args.X_pump, args.beta, args.gamma1, args.gamma2, args.rho, args.mu, args.alphaTilt, args.power, args.kapa, args.Lam, args.DTilt, args.DTilt_n
    # arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)
    # y_tmp = odeint(dynamics, state, t_span, args=(0, arrgs,))
    # print(y_tmp)
    state_buf = np.zeros((0, 500))

    for t in range(env.t_nums - 1):
        action = 0
        state_next, reward, done, info = env.step(action)
        state_buf = np.vstack((state, state_buf))
        state = state_next
        print(t, state)
    state_buf = np.vstack((state, state_buf))