####################################################
# Description: Wound Control
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-12-22
####################################################

import os
import copy
import time
from datetime import datetime

from pathlib import Path
import numpy as np
import pandas as pd
import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from algs.deepmapper import DeepMapper
from algs.dqn import Agent_DDQN


class AlphaHeal(object):

    def __init__(self, deviceArgs):
        super(AlphaHeal, self).__init__()
        # hyper-parameter setting for low level control device
        self.deviceArgs = deviceArgs
        self.wound_num = self.deviceArgs.wound_num
        self.writer = SummaryWriter(log_dir=self.deviceArgs.runs_device)
        self.delta_t = self.deviceArgs

        self.ch_map = {'{}'.format(idx + 1): idx for idx in range(self.deviceArgs.n_chs)}
        # TODO: Mapping from integer to string wound index
        self.wound_no_2_abc = {1: 'A', 2: 'B', 3: 'C',
                               5: 'D', 6: 'E', 7: 'F'}

        self.max_prolif = 0.0

        # Timers
        self.additional_time = 0

        # control counter
        self.ctrstep = 0

        # previous and current control signal for device
        self.ref = self.ref_prev = 0.0
        # timer and flag for open loop control
        self.ref_set_time = None
        self.ref_set_flag = False

        # deep learning models
        self.mapper = DeepMapper(deviceArgs=deviceArgs, writer=self.writer)
        self.rlagent = Agent_DDQN(deviceArgs=deviceArgs, writer=self.writer)
        self.state_cur = self.state_next = np.array([1, 0, 0, 0])
        self.act_space = np.linspace(0, self.deviceArgs.maxCurrent, self.rlagent.action_size)
        self.eps = self.rlagent.args.eps_start
        self.rlcnt = 0

        self.all_im_dirs_len_pre = 0

    def set_var(self):
        zeros_ch = input("Please select channels that will be set to 0. "
                         "\n For example, if you want channel 1,2,3 to 0, then input 1, 2, 3 "
                         "\n If no channel selected, just hit Enter"
                         "\n Zero Channels: ")
        if len(zeros_ch) > 0:
            # remove spaces
            zeros_ch = zeros_ch.replace(' ', '')
            # split according to ,
            zeros_ch = zeros_ch.split(',')
            # map to channels starting from 0
            self.zeros_ch = [self.ch_map[zc] for zc in zeros_ch]
        else:
            self.zeros_ch = []

        hours = input('Please set duration of experiment. For example: 12'
                      '\n Exp Duration: ')  # total run time in hours
        self.hours = int(hours)

        if not self.deviceArgs.close_loop:
            self.ref = self.ref_prev = self.deviceArgs.low_target

        print('Experiment will be run for {} hrs!!!'.format(self.hours))

    def read_csv(self, file_name):
        read_flag = 0
        df = None
        while read_flag == 0:
            try:
                df = pd.read_csv(file_name)
                read_flag = 1
            except:
                time.sleep(0.1)
                print("The {} was not readable, trying again to read the file".format(file_name))
        return df

    def save_csv(self, start_time):

        '''

        @param start_time: start time
        @return: numpy array containing all the target current
        '''

        time_hour = (time.time() - start_time) / 3600
        csv_table = {'t(s)': [time_hour]}
        csv_table |= {'target_I_ch{}(muA)'.format(ich): [self.ref]
                      for ich in range(self.deviceArgs.n_chs)}
        # set those channels to always output zero current
        for ich in self.zeros_ch:
            csv_table['target_I_ch{}(muA)'.format(ich)][0] = 0.0

        target_current = np.array([csv_table['target_I_ch{}(muA)'.format(ich)][0]
                                   for ich in range(self.deviceArgs.n_chs)])
        dataFrame = pd.DataFrame(csv_table)

        if self.ctrstep == 0:
            dataFrame.to_csv(self.deviceArgs.target_current_file_name, index=False)
        else:
            dataFrame.to_csv(self.deviceArgs.target_current_file_name, mode='a', index=False, header=False)

        return target_current

    def plot(self):
        '''

        @return:
        '''

        target_currents = [self.ref for _ in range(self.deviceArgs.n_chs)]
        for ich in range(self.deviceArgs.n_chs):
            if ich in self.zeros_ch:
                target_currents[ich] = 0

        self.writer.add_scalars('/currents/wound_{}'.format(self.wound_num),
                                {'tc_ch_{}'.format(ich): target_currents[ich]
                                 for ich in range(self.deviceArgs.n_chs)}, self.ctrstep)

    def open_loop(self, ct, wds, df_title):
        '''

        @param ct: (float) current time
        @param wds: (tuple) wound stages containing hemostasis, inflammation,
                                                    proliferation, maturation
        @param df_title: (string) title of data frame
        @return: treatment actuation for device
        '''
        ph, pi, pp, pm = wds

        # TODO: This need to be checked before experiment
        wound_no_abc_name = "Wound_{}".format(self.wound_no_2_abc[self.wound_num])
        wound_no_abc_name = wound_no_abc_name.lower()

        wound_no_int_name = "Wound_{}".format(self.wound_num)
        wound_no_int_name = wound_no_int_name.lower()


        # update max prob of proliferation
        self.max_prolif = max(pp, self.max_prolif)

        if wound_no_abc_name in df_title.lower() or wound_no_int_name in df_title.lower():
            if (self.max_prolif > self.deviceArgs.open_trigger) and \
                    self.ref == self.deviceArgs.low_target:
                self.ref_prev = copy.deepcopy(self.ref)
                self.ref = self.deviceArgs.heal_target
                self.ref_set_flag = True

            if self.ref_set_flag:
                self.ref_set_time = ct
                self.ref_set_flag = False
            if self.ref_set_time is not None and \
                    (time.time() - self.ref_set_time) >= 6 * 3600:
                # after 6 hours, we set a higher current
                self.ref_prev, self.ref = self.ref, self.deviceArgs.high_target

    def closed_loop(self):
        '''
        Closed loop control using deep reinforcement learning
        @return:
        '''

        # first check if new images been added to the folder
        root_images_dir = self.deviceArgs.device_im_dir + 'Wound_{}/'.format(self.deviceArgs.wound_num)

        try:
            all_im_dirs = os.listdir(root_images_dir)
            if len(all_im_dirs) > self.all_im_dirs_len_pre:
                sorted_im_dir = sorted(all_im_dirs)
                if sorted_im_dir[-1].startswith('20') and len(os.listdir(root_images_dir + sorted_im_dir[-1])) > 0:
                    # TODO: Disable in the future
                    dirtmp = root_images_dir + sorted_im_dir[-1]
                    self.state_next = self.mapper.ws_est(dirtmp)
                    pics_dir = []

                    for tmp in sorted(os.listdir(self.mapper.dsmg_dir)):
                        print(tmp[:16], sorted_im_dir[-1])
                        if tmp.startswith('20'):
                            if tmp[:16] <= sorted_im_dir[-1]:
                                pics_dir.append(self.mapper.dsmg_dir + tmp)
                            else:
                                break

                    self.mapper.test(self.rlcnt, pics_dir)
                    # self.state_next = self.mapper.ws_est(dirtmp)
                    reward = -1 if self.state_next[-1] < 0.95 else 0
                    done = False if self.state_next[-1] < 0.95 else True
                    action = self.act_space[self.rlagent.act(self.state_cur, eps=self.eps)]
                    if self.state_next[0] >= 0.9 or self.state_next[0] <= 0.1:
                        action = 0.0
                    self.rlagent.step(self.state_cur, action, reward, self.state_next, done)
                    # update next state
                    self.state_cur = copy.deepcopy(self.state_next)
                    self.ref_prev = self.ref
                    self.ref = action
                    self.eps = max(self.rlagent.args.eps_end, self.rlagent.args.eps_decay * self.eps)  # decrease epsilon
                    print('On {} \t Wound Prob: {} \t Actuation: {:.4f} TL: {}'.format(dirtmp.split('/')[-1], self.state_cur, action, len(pics_dir)))

                    self.all_im_dirs_len_pre = len(all_im_dirs)
                    self.rlcnt += 1

        except:
            print('unable to read images, try next time')

    def control(self):

        # first, we need to set some hyper-parameters from outside command
        self.set_var()

        # TODO: let the party begin

        # current healnet prob table size
        m_size_old_im = len(self.read_csv(self.deviceArgs.healnet_prob_file_name).index)
        # feed back from device
        last_modified_time = current_modified_time = os.path.getmtime(self.deviceArgs.prabhat_cv_file_name)

        # timers time
        start_time = time.time()
        last_sample_time = 0.0

        # counting the number of healnet predictions
        hpcnter = 0

        while True:
            if last_sample_time >= (self.hours * 3600) or \
                    ((time.time() - start_time) > (self.hours * 3600 + self.additional_time)):
                print("The code has been run {:.4f} hours".format((time.time() - start_time) / 3600))
                break

            # read healhel predictions
            df_im = self.read_csv(self.deviceArgs.healnet_prob_file_name)
            m_size_im = len(df_im.index)

            # apply treatment according to healnet predictions
            # If there is new predictions from healnet
            for j in range(m_size_im - m_size_old_im):
                df_title = df_im.iat[j + m_size_old_im, 0]
                ph = float(df_im.iat[j + m_size_old_im, 4])
                pi = float(df_im.iat[j + m_size_old_im, 5])
                pp = float(df_im.iat[j + m_size_old_im, 6])
                pm = float(df_im.iat[j + m_size_old_im, 7])

                self.writer.add_scalars('/HealNetPred/wound_{}'.format(self.wound_num),
                                        {'hemostasis': ph,
                                         'inflammation': pi,
                                         'proliferation': pp,
                                         'maturation': pm},
                                        hpcnter)
                hpcnter += 1

                if not self.deviceArgs.close_loop:
                    # open loop control
                    self.open_loop(time.time(), (ph, pi, pp, pm), df_title)

            if self.deviceArgs.close_loop:
                # close loop control
                self.closed_loop()
            # save control info to csv
            self.save_csv(start_time)

            # update healnet prob table size
            m_size_old_im = m_size_im

            # time.sleep(3600)

            # Each time there's new image, control
            print("Waiting for new wound images...")
            img_dir = self.deviceArgs.device_im_dir + 'Wound_{}/'.format(self.deviceArgs.wound_num)
            old_es_len = len(os.listdir(img_dir))
            while len(os.listdir(img_dir)) <= old_es_len:
                time.sleep(0.1)
            print("New wound images found. Start control...")
            # # check file modification without reading it
            # current_time = time.time()
            # print("Waiting for actuator to respond")
            # while (current_modified_time <= last_modified_time):
            #     current_modified_time = os.path.getmtime(self.deviceArgs.prabhat_cv_file_name)
            #     time.sleep(0.1)
            # # The following is needed in case there is communication or power failure for more than 60 seconds
            # response_time = (time.time() - current_time)
            # if (response_time > self.deviceArgs.response_time_threshold):
            #     self.additional_time += (response_time - self.deviceArgs.response_time_threshold)
            #     print("Additional time is {:.4f} hours".format(self.additional_time / 3600))
            # print("Actuator responded")
            # last_modified_time = current_modified_time
            # m_size = 0
            # prbs_df = None
            # print("waiting for new data")
            # while m_size == 0:
            #     prbs_df = self.read_csv(self.deviceArgs.prabhat_cv_file_name)
            #     m_size = len(prbs_df.index)
            #     time.sleep(0.5)
            # last_sample_time = prbs_df.iat[-1, 0]

            # plots
            self.plot()

            # we have finish one-step control
            self.ctrstep += 1