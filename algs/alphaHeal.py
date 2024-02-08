<<<<<<< Updated upstream
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
=======
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

import collections

from PIL import Image
from keras.utils import img_to_array

from envs.env import simple
from scipy.integrate import odeint

import torch.nn.functional as F

import sys

class OpCSVProb(object):

    def __init__(self, Estimator):
        self.prob_table_path = 'wound_probs_{}.csv'.format(Estimator)
        try:
            self.prob_table = pd.read_csv(self.prob_table_path)
        except:
            headers = {"Image": [], "Time Processed": [], "Blur": [], "Patches": [], "Hemostasis": [],
                       "Inflammation": [], "Proliferation": [], "Maturation": []}

            table = pd.DataFrame.from_dict(headers)
            table.to_csv(self.prob_table_path, index=False)
            self.prob_table = pd.read_csv(self.prob_table_path)

    def save_2_csv(self, image, probs):
        self.prob_table.loc[len(self.prob_table)] = [str(image), time.time(), 0, len(probs),
                                                     probs[0], probs[1], probs[2], probs[3]]
        self.prob_table.to_csv(self.prob_table_path, index=False)



class SubSummaryWriter(object):

    def __init__(self, writer):
        self.writer = writer

        self.org_im_buf = []
        self.gen_im_buf = []

        self.probs_deepmapper = np.empty(4)

    def _sqEv(self, org, gen, ss=5):
        self.org_im_buf.append(org)
        self.gen_im_buf.append(gen)

        numIms = len(self.org_im_buf)
        numRows = numIms // ss
        if numIms % ss > 0:
            numRows += 1

        dst1 = Image.new('RGB', (128 * ss, 128 * numRows))
        dst2 = Image.new('RGB', (128 * ss, 128 * numRows))
        for j in range(numRows):
            for i in range(ss):
                if (i + j * ss) < numIms:
                    org_im_tmp = self.org_im_buf[i + j * ss]
                    gen_im_tmp = self.gen_im_buf[i + j * ss]
                    org_im_tmp = org_im_tmp * 255
                    gen_im_tmp = gen_im_tmp * 255
                    org_im_tmp = Image.fromarray(org_im_tmp.astype(np.uint8))
                    gen_im_tmp = Image.fromarray(gen_im_tmp.astype(np.uint8))
                    dst1.paste(org_im_tmp, (i * 128, 128 * j))
                    dst2.paste(gen_im_tmp, (i * 128, 128 * j))
        return img_to_array(dst1) / 255.0, img_to_array(dst2) / 255.0

    def gen_org_im_plot(self, org, gen, cnter):
        dst_org, dst_gen = self._sqEv(org, gen)
        self.writer.add_image('wsd_stage_deepmapper/orgs', dst_org, cnter, dataformats='HWC')
        self.writer.add_image('wsd_stage_deepmapper/gens', dst_gen, cnter, dataformats='HWC')

    def probs_plot(self, probs, cnters, info, wound_num):
        self.writer.add_scalars('/{}/wound_{}'.format(info, wound_num),
                                {'hemostasis': probs[0],
                                 'inflammation': probs[1],
                                 'proliferation': probs[2],
                                 'maturation': probs[3]},
                                cnters[info])

    def probs_deepmapper_plot(self, probs, y_tmp, xrange, cnt):
        self.probs_deepmapper = np.vstack((self.probs_deepmapper, probs))
        xrange_cur = xrange[:min(len(self.probs_deepmapper) - 1, len(y_tmp))]

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot()
        ax.scatter(xrange_cur, self.probs_deepmapper[1:len(y_tmp), 0], color='r')  # , label='Hemostasis')
        ax.scatter(xrange_cur, self.probs_deepmapper[1:len(y_tmp), 1], color='g')  # , label='Inflammation')
        ax.scatter(xrange_cur, self.probs_deepmapper[1:len(y_tmp), 2], color='b')  # , label='Proliferation')
        ax.scatter(xrange_cur, self.probs_deepmapper[1:len(y_tmp), 3], color='y')  # , label='Maturation')

        ax.plot(xrange, y_tmp[:, 0], color='r', label='H')
        ax.plot(xrange, y_tmp[:, 1], color='g', label='I')
        ax.plot(xrange, y_tmp[:, 2], color='b', label='P')
        ax.plot(xrange, y_tmp[:, 3], color='y', label='M')

        leg_pos = (1, 0.5)
        ax.legend(loc='center left', bbox_to_anchor=leg_pos)
        ax.set_xlabel('Time (day)')

        self.writer.add_figure('wsd_stage_deepmapper/prob', fig, cnt)


class AlphaHeal(object):

    def __init__(self, deviceArgs):
        super(AlphaHeal, self).__init__()

        # hyper-parameter setting for low level control device
        self.deviceArgs = deviceArgs
        self.wound_num = self.deviceArgs.wound_num
        self.delta_t = self.deviceArgs

        # setting total time
        self.set_var()

        self.probs_csv_deepmapper = OpCSVProb('DeepMapper')

        self.ch_map = {'{}'.format(idx + 1): idx for idx in range(self.deviceArgs.n_chs)}
        # TODO: Mapping from integer to string wound index
        self.wound_no_2_abc = {1: 'A', 2: 'B', 3: 'C',
                               5: 'D', 6: 'E', 7: 'F'}
        self.max_prolif = 0.0
        # Timers
        self.additional_time = 0

        # previous and current control signal for device
        self.ref = self.ref_prev = 0.0
        # timer and flag for open loop control
        self.ref_set_time = None
        self.ref_set_flag = False

        self.writer = SummaryWriter(log_dir=self.deviceArgs.runs_device)
        self.imwriter = SubSummaryWriter(self.writer)
        # deep learning models
        self.mapper = DeepMapper(deviceArgs=deviceArgs, writer=self.writer)
        self.rlagent = Agent_DDQN(deviceArgs=deviceArgs, writer=self.writer)
        self.state_cur = self.state_next = np.array([1, 0, 0, 0])
        self.act_space = np.linspace(0, self.deviceArgs.maxCurrent, self.rlagent.action_size)
        self.eps = self.rlagent.args.eps_start
        self.rlcnt = 0
        self.image_time_stamp = list()

        self.cnters = {
            'HealNet_Prediction': 0,
            'DeepMapper_Prediction': 0,
            'control_step': 0
        }

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

        hours = input('Please set duration (hour) of experiment. For example: 12'
                      '\n Exp Duration: hours')  # total run time in hours
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

    def read_gui(self):
        try:
            shut_off = pd.read_csv('../Interface_code/shutoff.csv')
            # drug_coc = pd.read_csv('../drug_concentration_data.csv')
            # efff_coc = pd.read_csv('../drug_concentration_data.csv')
            if shut_off['Current'].iloc[0] == 1:
                return True
        except:
            print("No setting from GUI")
            return False

    def save_csv_control(self, start_time):

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
        try:
            dataFrame.to_csv(self.deviceArgs.target_current_file_name, mode='a', index=False, header=False)
        except:
            dataFrame.to_csv(self.deviceArgs.target_current_file_name, index=False)

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
                                 for ich in range(self.deviceArgs.n_chs)}, self.cnters['control_step'])

    def pred_deepmapper(self):
        '''
        Wound stage estimation using deepmapper
        '''
        # first check if new images been added to the folder
        root_images_dir = self.deviceArgs.device_im_dir + 'Wound_{}/'.format(self.deviceArgs.wound_num)
        for img_dir in os.listdir(root_images_dir):
            # 1. it's a valid director? 2. does it contains images?
            if str(root_images_dir + img_dir) not in list(self.probs_csv_deepmapper.prob_table["Image"]):
                if img_dir.startswith('2024') and len(os.listdir(root_images_dir + img_dir)) > 1:
                    if img_dir not in self.image_time_stamp:
                        try:
                            probs = collections.defaultdict(list)
                            wst_deepmapper, org_im, gen_im = self.mapper.ws_est_gen(root_images_dir + img_dir)
                            probs[img_dir].append(wst_deepmapper)
                            self.image_time_stamp.append(img_dir)
                            avg_probs = np.array([np.mean(prob_tmp, axis=0) for prob_tmp in probs.values()]).mean(axis=0)

                            y_opt = odeint(simple, np.array([1., 0., 0., 0.]), self.mapper.xrange,
                                           args=(np.array([F.sigmoid(self.mapper.model.Kh).data.cpu().numpy()[0],
                                                           F.sigmoid(self.mapper.model.Ki).data.cpu().numpy()[0],
                                                           F.sigmoid(self.mapper.model.Kp).data.cpu().numpy()[0]]),))


                            self.imwriter.probs_deepmapper_plot(avg_probs, y_opt, self.mapper.xrange, self.cnters["DeepMapper_Prediction"])
                            self.imwriter.gen_org_im_plot(org_im, gen_im, self.cnters["DeepMapper_Prediction"])
                            self.cnters["DeepMapper_Prediction"] += 1
                            self.probs_csv_deepmapper.save_2_csv(root_images_dir + img_dir, avg_probs)
                            # yield avg_probs
                        except:
                            print('unable to read {} images, try next time'.format(img_dir))
                            # yield None

    def _wst(self, m_size_old_im, estimator):
        '''
        Wound stage estimation using healnet
        '''
        # read healhel predictions
        if estimator == 'DeepMapper':
            df_im = self.read_csv(self.deviceArgs.deepmapper_prob_file_name)
        else:
            df_im = self.read_csv(self.deviceArgs.healnet_prob_file_name)
        m_size_im = len(df_im.index)

        # TODO: This need to be checked before experiment
        wound_no_abc_name = "Wound_{}".format(self.wound_no_2_abc[self.wound_num])
        wound_no_abc_name = wound_no_abc_name.lower()

        wound_no_int_name = "Wound_{}".format(self.wound_num)
        wound_no_int_name = wound_no_int_name.lower()

        # apply treatment according to healnet predictions
        # If there is new predictions from healnet
        probs, pcnter = [0] * 4, 1
        for j in range(m_size_im - m_size_old_im):
            df_title = df_im.iat[j + m_size_old_im, 0]
            if wound_no_abc_name in df_title.lower() or wound_no_int_name in df_title.lower():
                # Monter Carlo Average Estimation
                for pidx in range(4):
                    probs[pidx] += (float(df_im.iat[j + m_size_old_im, 4 + pidx]) - probs[pidx]) / pcnter
                pcnter += 1
                if estimator == 'DeepMapper':
                    self.imwriter.probs_plot(probs, self.cnters, 'DeepMapper_Prediction', self.wound_num)
                else:
                    self.imwriter.probs_plot(probs, self.cnters, 'HealNet_Prediction', self.wound_num)
                self.cnters['HealNet_Prediction'] += 1

        return np.array(probs), m_size_im

    def open_loop(self, wds):
        '''

        @param ct: (float) current time
        @param wds: (tuple) wound stages containing hemostasis, inflammation,
                                                    proliferation, maturation
        @param df_title: (string) title of data frame
        @return: treatment actuation for device
        '''
        ct = time.time()
        ph, pi, pp, pm = wds

        # update max prob of proliferation
        self.max_prolif = max(pp, self.max_prolif)
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

    def closed_loop(self, new_state):
        '''
        Closed loop control using deep reinforcement learning
        @return:
        '''

        self.state_next = copy.deepcopy(new_state)
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
        print('Wound Prob: {} \t Actuation: {:.4f}'.format(self.state_cur, action))
        self.rlcnt += 1

    def control(self):
        # first, we need to set some hyper-parameters from outside command
        # TODO: let the party begin
        # current healnet prob table size

        try:
            m_size_old_im_healnet = len(self.read_csv(self.deviceArgs.healnet_prob_file_name).index)
        except:
            m_size_old_im_healnet = 0

        try:
            m_size_old_im_deepmapper = len(self.read_csv(self.deviceArgs.deepmapper_prob_file_name).index)
        except:
            m_size_old_im_deepmapper = 0

        # feed back from device
        # timers time
        start_time = time.time()
        last_sample_time = 0.0

        while True:
            if last_sample_time >= (self.hours * 3600) or \
                    ((time.time() - start_time) > (self.hours * 3600 + self.additional_time)):
                print("The code has been run {:.4f} hours".format((time.time() - start_time) / 3600))
                break

            if self.deviceArgs.close_loop:
                # close loop control
                self.pred_deepmapper()
                deepmapper_probs, m_size_old_im_deepmapper = self._wst(m_size_old_im_deepmapper, "DeepMapper")
                self.closed_loop(deepmapper_probs)
            else:
                healnet_probs, m_size_old_im_healnet = self._wst(m_size_old_im_healnet, "HealNet")
                self.open_loop(healnet_probs)
            # constantly check gui output
            # self.read_gui()
            # save control info to csv
            if not self.read_gui():
                self.save_csv_control(start_time)

            # Each time there's new image, control
            print("Waiting for new wound images...")
            img_dir = self.deviceArgs.device_im_dir + 'Wound_{}/'.format(self.deviceArgs.wound_num)
            old_es_len = len(os.listdir(img_dir))
            while len(os.listdir(img_dir)) <= old_es_len:
                if self.read_gui():
                    self.ref_prev = self.ref
                    self.ref = 0
                    self.save_csv_control(start_time)
                    print("Emergent Shut Down Performed!!!")
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
            self.cnters['control_step'] += 1
>>>>>>> Stashed changes
