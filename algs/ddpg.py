import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import namedtuple, deque

import random

from scipy.integrate import odeint
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

from cfgs.config import GetParameters


def simple(z, t, action):
    kh, ki, kp = action.clip(0.0, 1.0)
    H, I, P, M = z

    dhdt = -kh * H
    didt = kh * H - ki * I
    dpdt = ki * I - kp * P
    dmdt = kp * P

    dzdt = [dhdt, didt, dpdt, dmdt]
    return dzdt

class SimpleEnv(object):

    def __init__(self):
        super(SimpleEnv, self).__init__()

        self.action_size = 3
        self.state_size = 4

        self.t_days = 60
        self.t_nums = 301
        self.t_span = np.linspace(0, self.t_days, self.t_nums)

        self.k_opt = np.array([0.5, 0.3, 0.1])
        self.X_noise = None
        self.state_init = np.array([1, 0, 0, 0])
        self.state = self.y0 = self.state_init
        self.cnter = 0

    def ode_solver(self, action, FT=False):
        if not FT:
            tspan_tmp = [self.t_span[self.cnter], self.t_span[self.cnter + 1]]
            y_tmp = odeint(simple, self.y0, tspan_tmp, args=(action,))
            self.y0 = y_tmp[1]
            X = np.array(y_tmp[1, :])
        else:
            tspan_tmp = self.t_span
            y_tmp = odeint(simple, self.y0, tspan_tmp, args=(action,))
            X = np.array(y_tmp)
        return X

    def step(self, action):
        X = self.ode_solver(action)
        next_state = X.reshape(-1)

        self.cnter += 1
        next_state_real = self.X_noise[self.cnter, :]
        self.state = next_state

        reward = -np.sqrt((next_state_real[0] - next_state[0]) ** 2 +
                          (next_state_real[1] - next_state[1]) ** 2 +
                          (next_state_real[2] - next_state[2]) ** 2 +
                          (next_state_real[3] - next_state[3]) ** 2)
        info = [self.state, 1]
        return self.state, reward, self.done, info

    def reset(self):
        self.state = self.y0 = self.state_init
        self.done = False
        self.cnter = 0

        self.X_noise = self.ode_solver(self.k_opt, FT=True)

        return self.state

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.999  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 3e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0001  # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        return F.sigmoid(self.fc2(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(env.state_size, env.action_size, random_seed).to(device)
        self.actor_target = Actor(env.state_size, env.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(env.state_size, env.action_size, random_seed).to(device)
        self.critic_target = Critic(env.state_size, env.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(env.action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(env.action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, 0, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def ddpg():
    args = GetParameters()
    args.alg_rl = 'ddpg'
    args.model_dir = '../../../res_wound_rl/res/models/models_{}/'.format(args.alg_rl)
    args.data_dir = '../../../res_wound_rl/res/data/data_{}/'.format(args.alg_rl)
    args.figs_dir = '../../../res_wound_rl/res/figs/figs_{}/'.format(args.alg_rl)

    dirs = [args.model_dir, args.data_dir, args.figs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    env = SimpleEnv()
    agent = Agent(env=env, random_seed=10)

    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, args.n_episodes + 1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(env.t_nums - 1):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('Algm: {} Ep: {}/{}\tAvgS: {:.2f}, info: [M: {:.2f} Stage: {}] Action: [{:.2f}, {:.2f}, {:.2f}]'.format(
            args.alg_rl, i_episode, args.n_episodes, np.mean(scores_deque), info[0][3], info[1], action[0], action[1], action[2]), flush=True)

        if i_episode % 5 == 0:
            torch.save(agent.actor_local.state_dict(),
                       args.model_dir + 'checkpoint_actor_anum_{}_ep_{}.pth'.format(env.action_size, i_episode))
            torch.save(agent.critic_local.state_dict(),
                       args.model_dir + 'checkpoint_critic_anum_{}_ep_{}.pth'.format(env.action_size, i_episode))
    return scores


if __name__ == "__main__":
    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()