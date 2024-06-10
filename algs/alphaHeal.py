####################################################
# Description: Wound Control
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-12-22
####################################################

import os
import copy
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from algs.deepmapper import DeepMapper
from algs.dqn import Agent_DDQN

import collections

from PIL import Image
from keras.utils import img_to_array

from envs.env import simple
from scipy.integrate import odeint

import torch.nn.functional as F

from tqdm import tqdm

from utils.tools import str2sec

from sklearn.tree import DecisionTreeRegressor

# For Drug concentration vs time
Faraday = 96485.3321   # Faraday constant
eta = 0.2        # Pump efficiency
g_per_mol = 309.33  # Molecular weight of flx


def drug_conc(current, time):
    if current < 0:
        current = 0.0
    charge = current * time
    dose = eta * charge * g_per_mol / (Faraday * 1e3)

    return dose

class OpCSVProb(object):

    def __init__(self, Estimator, args):
        if Estimator == "DeepMapper":
            self.prob_table_path = args.deepmapper_prob_file_name
        else:
            self.prob_table_path = args.healnet_prob_file_name

        try:
            self.prob_table = pd.read_csv(self.prob_table_path)
        except:
            headers = {"Image": [], "Time Processed": [], "Blur": [], "Patches": [], "Hemostasis": [],
                       "Inflammation": [], "Proliferation": [], "Maturation": [], "Progress": [], 'Treatment': []}

            table = pd.DataFrame.from_dict(headers)
            table.to_csv(self.prob_table_path, index=False)
            self.prob_table = pd.read_csv(self.prob_table_path)

        # Get previous prob time
        if len(self.prob_table['Time Processed']) > 0:
            self.prev_prob_time = self.prob_table['Time Processed'].iat[-1]
        else:
            self.prev_prob_time = None

    def save_2_csv(self, image, probs, progs, treat):
        # self.prev_prob_time = time.time()
        tt = 'EF' if treat else 'Flx'
        # TODO: TO Remove
        probs[1] -= 0.007
        probs = np.clip(probs, 0, 1.0)
        self.prob_table.loc[len(self.prob_table)] = [str(image), time.time(), 0, len(probs),
                                                     probs[0], probs[1], probs[2], probs[3], progs, tt]
        # self.prob_table.loc[len(self.prob_table)] = [str(image), time.time(), 0, len(probs),
        #                                              probs[0], probs[1], probs[2], probs[3], progs, tt]
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
        self.ch_map = {'{}'.format(idx + 1): idx + 1 for idx in range(self.deviceArgs.n_chs)}

        # setting total time
        self.set_var()

        self.shut_off_flag = False
        self.new_wound_stage_flag = True
        self.wound_stage_prev = np.array([1, 0, 0, 0])
        self.probs_csv_deepmapper = OpCSVProb('DeepMapper', self.deviceArgs)

        # TODO: Mapping from integer to string wound index

        map_df = pd.read_csv(self.deviceArgs.mapping_table_dir)
        self.wound_no_2_abc = {}
        for i in range(len(map_df)):
            self.wound_no_2_abc |= {int(map_df['Wound No.'].iloc[i]): map_df['Camera Identifier'].iloc[i].split('_')[-1]}
        print('Mapping Table Set To: {}'.format(map_df))
        self.max_prolif = 0.0
        # Timers
        self.additional_time = 0

        # reference channel value
        self.zero_act = -1.0

        # previous and current control signal for device
        self.ref = self.ref_prev = self.zero_act
        # timer and flag for open loop control
        self.ref_set_time = None
        self.ref_set_flag = False

        self.writer = SummaryWriter(log_dir=self.deviceArgs.runs_device)
        self.imwriter = SubSummaryWriter(self.writer)
        # deep learning models
        self.mapper = DeepMapper(deviceArgs=deviceArgs, writer=self.writer)
        self.rlagent = Agent_DDQN(deviceArgs=deviceArgs, writer=self.writer)
        self.state_cur = self.state_next = np.array([1, 0, 0, 0])
        self.act_space = np.linspace(self.deviceArgs.minCurrent, self.deviceArgs.maxEFCurrent, self.rlagent.action_size)
        self.act_EF_space = np.linspace(self.deviceArgs.minCurrent, self.deviceArgs.maxEFCurrent, self.rlagent.action_size)
        self.act_Flx_space = np.linspace(self.deviceArgs.minCurrent, self.deviceArgs.maxFlxCurrent, self.rlagent.action_size)
        # TODO: No zero current will be set
        # if self.deviceArgs.treatment != 1:
        #     self.act_space[0] = self.zero_act
        self.eps = self.rlagent.args.eps_start
        self.rlcnt = 0
        self.image_time_stamp = list()
        self.wp_prev = self.wp_cur = 0

        self.drug_duration_pre = 0.0
        self.drug_total = 0.0
        self.time_total = 0.0
        # timer for Flx delivery within 2hour time window
        self.time_inbetween = 0.0
        self.drug_inbetween_total = 0.0
        self.drug_inbetween_flag = False
        self.drug_max_flag = False

        self.emergent_shut_off_set_flag = False

        self.cnters = {
            'HealNet_Prediction': 0,
            'DeepMapper_Prediction': 0,
            'control_step': 0
        }
        self.init_progressor()
        self.treatmentEF = True

    def init_progressor(self):
        self.xrange = np.linspace(0, 20, 240)
        self.y_opt = odeint(simple, np.array([1., 0., 0., 0.]), self.xrange,
                       args=(np.array([F.sigmoid(self.mapper.model.Kh).data.cpu().numpy()[0],
                                       F.sigmoid(self.mapper.model.Ki).data.cpu().numpy()[0],
                                       F.sigmoid(self.mapper.model.Kp).data.cpu().numpy()[0]]),))
        # create a regressor object
        self.progressor = DecisionTreeRegressor(random_state=0)
        # fit the regressor with X and Y data
        self.progressor.fit(self.y_opt, self.xrange)
        print("Finish Initializing Wound Progression Prediction")

    def set_var(self):
        # TODO: Exp23 Check
        # zeros_ch = input("Please select channels that will be set to 0. "
        #                  "\n For example, if you want channel 1,2,3 to 0, then input 1, 2, 3 "
        #                  "\n If no channel selected, just hit Enter"
        #                  "\n Zero Channels: ")
        zeros_ch = ""

        ef_ch = input("Please select EF channels , default is odd for EF and even for Flx"
                         "\n For example, if you want channel 1,2,3 to deliver EF, then input 1, 2, 3 "
                         "\n EF Channels: ")

        if len(ef_ch) > 0:
            try:
                # remove spaces
                ef_ch = ef_ch.replace(' ', '')
                # split according to ,
                ef_ch = ef_ch.split(',')
                # map to channels starting from 0
                self.ef_ch = [self.ch_map[efc] for efc in ef_ch]
                self.flx_ch = list(set(self.ch_map.values()) - set(self.ef_ch))
                print("EF Channels: {} \t Flx Channels: {}".format(self.ef_ch, self.flx_ch))
            except:
                print("Please Set EF Channels!!!")
        else:
            self.ef_ch = [1, 3, 5, 7]
            self.flx_ch = list(set(self.ch_map.values()) - set(self.ef_ch))
            print("EF Channels: {} \t Flx Channels: {}".format(self.ef_ch, self.flx_ch))
            assert True, print("Please Set EF/Flx Channels!!!")

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
                      '\n Exp Duration: hours: ')  # total run time in hours
        self.hours = float(hours)

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

    def read_csv_imediate(self, file_name):
        df = None
        try:
            df = pd.read_csv(file_name)
        except:
            print("The {} was not readable, trying again to read the file".format(file_name))
        return df

    def read_gui(self):
        try:
            shut_off = pd.read_csv('shutoff.csv')
            # drug_coc = pd.read_csv('../drug_concentration_data.csv')
            # efff_coc = pd.read_csv('../drug_concentration_data.csv')
            if shut_off.Current.iloc[0] == 0:
                print("Emergent Shutoff Performed!!!")
                self.ref_prev, self.ref = self.ref, self.zero_act
                self.shut_off_flag = True
                self.new_wound_stage_flag = True
                return True
            else:
                return False
        except:
            # return
            # print("Reading settings from GUI Failed, try next time...")
            return False

    def save_csv_control(self, treat=None):

        '''

        @param start_time: start time
        @return: numpy array containing all the target current
        '''

        time_sec = time.time()
        csv_table = {'t(s)': [time_sec]}
        csv_table |= {'target_I_ch{}(muA)'.format(ich + 1): [self.ref]
                      for ich in range(self.deviceArgs.n_chs)}

        # set those channels to always output zero current
        # if treat is None:
        #     for ich in self.zeros_ch:
        #         csv_table['target_I_ch{}(muA)'.format(ich)][0] = self.zero_act
        # else:
        # current treatment is EF, set flx channel to zero
        if self.treatmentEF:
            for ich in self.flx_ch:
                csv_table['target_I_ch{}(muA)'.format(ich)][0] = 0.0
        # current treatment is Flx, set EF channel to zero
        else:
            for ich in self.ef_ch:
                csv_table['target_I_ch{}(muA)'.format(ich)][0] = self.zero_act

        target_current = np.array([csv_table['target_I_ch{}(muA)'.format(ich + 1)][0]
                                   for ich in range(self.deviceArgs.n_chs)])
        dataFrame = pd.DataFrame(csv_table)
        try:
            if os.path.exists(self.deviceArgs.target_current_file_name):
                dataFrame.to_csv(self.deviceArgs.target_current_file_name, mode='a', index=False, header=False)
            else:
                dataFrame.to_csv(self.deviceArgs.target_current_file_name, index=False)
        except:
            print('Saving Target File Failed, try next time!!!')

        return target_current

    def drug_const(self):
        try:
            df = pd.read_csv(self.deviceArgs.fl_comb_file_name)
            if len(df) >= 2:
                # cumulated time
                if df['time(s)'].iloc[-1] - df['time(s)'].iloc[-2] >= 0:
                    if not self.treatmentEF:
                        self.time_total += df['time(s)'].iloc[-1] - df['time(s)'].iloc[-2]

                    # TODO: Change to 24 hours
                    if self.time_total > 24.0 * 3600.0:
                        print('New day start!!!')
                        self.time_total = 0.0
                        self.drug_total = 0.0

                if self.drug_inbetween_total < 0.005:
                    self.drug_inbetween_flag = False
                    for ich in self.flx_ch:
                        if df['time(s)'].iloc[-1] - df['time(s)'].iloc[-2] >= 0:
                            self.drug_inbetween_total += drug_conc(df['dc_ch_{}'.format(ich)].iloc[-1],
                                                                   df['time(s)'].iloc[-1] - df['time(s)'].iloc[-2])

                # Drug exceed maximum dosage within one day
                else:
                    if not self.drug_inbetween_flag:
                        print('Exceed Drug Delivery Time Window; Shut off delivery until next batch!!!')
                        self.ref_prev, self.ref = self.ref, self.zero_act
                        # self.save_csv_control(start_time)
                        self.new_wound_stage_flag = True
                        self.drug_inbetween_flag = True

                if self.drug_total < self.deviceArgs.maxDosage:
                    self.drug_max_flag = False
                    for ich in self.flx_ch:
                        if df['time(s)'].iloc[-1] - df['time(s)'].iloc[-2] >= 0:
                            self.drug_total += drug_conc(df['dc_ch_{}'.format(ich)].iloc[-1],
                                                         df['time(s)'].iloc[-1] - df['time(s)'].iloc[-2])

                # Drug exceed maximum dosage within one day
                else:
                    if not self.drug_max_flag:
                        print('Exceed Drug Maximum; Shut off delivery until next day!!!')
                        self.ref_prev, self.ref = self.ref, self.zero_act
                        # self.save_csv_control(start_time)
                        self.new_wound_stage_flag = True
                        self.drug_max_flag = True

        except:
            print("Length of Drug not enough!")

    def plot(self):
        '''

        @return:
        '''

        ctime = time.time()
        device_currents = self.read_csv_imediate(self.deviceArgs.prabhat_cv_file_name)
        target_currents = self.read_csv_imediate(self.deviceArgs.target_current_file_name)
        wound_stages = self.read_csv_imediate(self.deviceArgs.deepmapper_prob_file_name)

        device_currents_list = [device_currents['I{} (uA)'.format(ich + 1)].iloc[-1] for ich in
                                range(self.deviceArgs.n_chs)]
        target_currents_list = [target_currents['target_I_ch{}(muA)'.format(ich + 1)].iloc[-1] for ich in
                                range(self.deviceArgs.n_chs)]

        treats = {0: 'Flx', 1: 'EF'}
        com_df = {'time(s)': [ctime]}
        com_df |= {'treat': [treats[int(self.treatmentEF)]]}
        com_df |= {
            'tc_ch_{}'.format(ich + 1): target_currents_list[ich] for ich in range(self.deviceArgs.n_chs)
        }
        com_df |= {
            'dc_ch_{}'.format(ich + 1): device_currents_list[ich] for ich in range(self.deviceArgs.n_chs)
        }

        if len(wound_stages) == 0:
            print("No Image files found!!!")

        image_name = os.listdir(wound_stages['Image'].iloc[-1])
        image_use = None
        for image_tmp in image_name:
            if image_tmp.startswith('2024') or image_tmp.startswith('2023'):
                image_use = image_tmp
                break

        com_df |= {
            'Image': wound_stages['Image'].iloc[-1] + '/' + image_use,
            'Hemostasis': wound_stages['Hemostasis'].iloc[-1],
            'Inflammation': wound_stages['Inflammation'].iloc[-1],
            'Proliferation': wound_stages['Proliferation'].iloc[-1],
            'Maturation': wound_stages['Maturation'].iloc[-1],
            'wound_progress_DRLctr': self.wp_cur,
            'cum_drug(mg/mol)': self.drug_total * 1000.0,
        }

        com_df = pd.DataFrame(com_df)
        if os.path.exists(self.deviceArgs.fl_comb_file_name):
            com_df.to_csv(self.deviceArgs.fl_comb_file_name, mode='a', index=False, header=False)
        else:
            com_df.to_csv(self.deviceArgs.fl_comb_file_name, index=False)

        self.writer.add_scalars('/target_currents/wound_{}'.format(self.wound_num),
                                {'tc_ch_{}'.format(ich + 1): target_currents_list[ich]
                                 for ich in range(self.deviceArgs.n_chs)}, self.cnters['control_step'])
        self.writer.add_scalars('/device_currents/wound_{}'.format(self.wound_num),
                                {'dc_ch_{}'.format(ich + 1): device_currents_list[ich]
                                 for ich in range(self.deviceArgs.n_chs)}, self.cnters['control_step'])
        self.writer.add_scalar('/drl/prog_wound_{}'.format(self.wound_num), self.wp_cur, self.cnters['control_step'])
        self.writer.add_scalar('/drl/drug_wound_{}'.format(self.wound_num), self.drug_total * 1000.0,
                               self.cnters['control_step'])

    def pred_deepmapper(self):
        '''
        Wound stage estimation using deepmapper
        '''
        # first check if new images been added to the folder
        to_abc = self.wound_no_2_abc[self.deviceArgs.wound_num]
        root_images_dir = self.deviceArgs.device_im_dir + 'Camera_{}/'.format(to_abc)

        print_image_exist = False
        while not os.path.exists(root_images_dir) or len(os.listdir(root_images_dir)) == 0:
            time.sleep(0.1)
            if not print_image_exist:
                print("Image Files Not Exist. Waiting for device images...")
                print_image_exist = True

        for img_dir in os.listdir(root_images_dir):
            if not img_dir.startswith('.'):
                # 1. it's a valid director? 2. does it contains images?
                if str(root_images_dir + img_dir) not in list(self.probs_csv_deepmapper.prob_table["Image"]):
                    # if img_dir.startswith('2024') and len(os.listdir(root_images_dir + img_dir)) > 1:
                    #     if img_dir not in self.image_time_stamp:
                    try:
                        # TODO: Add in-vivo
                        if self.deviceArgs.invivo:
                            # wait for 5mins until all images are stored
                            for _ in tqdm(range(300), desc='Waiting images...'):
                                time.sleep(1)
                        # Calculate the time different from last batch images

                        time_process = str2sec(str(img_dir))

                        if self.probs_csv_deepmapper.prev_prob_time is None:
                            self.probs_csv_deepmapper.prev_prob_time = time_process
                            time_dif = 0.0
                        else:
                            time_dif = time_process - self.probs_csv_deepmapper.prev_prob_time
                            self.probs_csv_deepmapper.prev_prob_time = time_process

                        # time_dif = time.time() - self.probs_csv_deepmapper.prev_prob_time
                        # self.probs_csv_deepmapper.prev_prob_time = time.time()

                        wst_deepmapper, org_im, gen_im = self.mapper.ws_est_gen(root_images_dir + img_dir, time_dif)
                        # self.image_time_stamp.append(img_dir)
                        # y_opt = odeint(simple, np.array([1., 0., 0., 0.]), self.mapper.xrange,
                        #                args=(np.array([F.sigmoid(self.mapper.model.Kh).data.cpu().numpy()[0],
                        #                                F.sigmoid(self.mapper.model.Ki).data.cpu().numpy()[0],
                        #                                F.sigmoid(self.mapper.model.Kp).data.cpu().numpy()[0]]),))
                        # self.imwriter.probs_deepmapper_plot(wst_deepmapper, y_opt, self.mapper.xrange, self.cnters["DeepMapper_Prediction"])
                        # self.imwriter.gen_org_im_plot(org_im, gen_im, self.cnters["DeepMapper_Prediction"])
                        # update wound progression only when new images arrive.
                        predict_date = self.progressor.predict(wst_deepmapper.reshape(1, -1))[0]
                        self.wp_prev, self.wp_cur = self.wp_cur, predict_date / 20.0

                        self.cnters["DeepMapper_Prediction"] += 1
                        self.probs_csv_deepmapper.save_2_csv(root_images_dir + img_dir, wst_deepmapper, self.wp_cur, self.treatmentEF)
                    except:
                        wst_deepmapper = [None] * 4
                        self.probs_csv_deepmapper.save_2_csv(root_images_dir + img_dir, wst_deepmapper, 0)
                        print('No Images found in {}, skip this time slot with None'.format(img_dir))

    def _wst(self, m_size_old_im, estimator):
        '''
        Wound stage estimation using healnet
        '''
        # TODO: Add Try Except
        # read healhel predictions
        if estimator == 'DeepMapper':
            df_im = self.read_csv(self.deviceArgs.deepmapper_prob_file_name)
        else:
            df_im = self.read_csv(self.deviceArgs.healnet_prob_file_name)
        m_size_im = len(df_im.index)

        # TODO: This need to be checked before experiment
        wound_no_abc_name = "Camera_{}".format(self.wound_no_2_abc[self.wound_num])
        wound_no_abc_name = wound_no_abc_name.lower()

        wound_no_int_name = "Camera_{}".format(self.wound_num)
        wound_no_int_name = wound_no_int_name.lower()

        # apply treatment according to healnet predictions
        # If there is new predictions from healnet
        probs, pcnter = np.zeros(4), 1
        for j in range(m_size_im - m_size_old_im):
            df_title = df_im.iat[j + m_size_old_im, 0]
            if wound_no_abc_name in df_title.lower() or wound_no_int_name in df_title.lower():
                # Monter Carlo Average Estimation
                for pidx in range(4):
                    if probs[pidx] is not None:
                        probs[pidx] += (float(df_im.iat[j + m_size_old_im, 4 + pidx]) - probs[pidx]) / pcnter
                pcnter += 1
                # self.imwriter.probs_plot(probs, self.cnters, '{}_Prediction'.format(estimator), self.wound_num)
                # self.cnters['{}_Prediction'.format(estimator)] += 1
        # in case there's no image data or all predictions are none
        if sum(probs) == 0:
            if m_size_im <= 0:
                probs = self.wound_stage_prev
                # self.probs_csv_deepmapper.save_2_csv('Initial Assumption', probs)
            elif np.any([(float(df_im.iat[-1, 4 + pidx])) for pidx in range(4)]) is None:
                probs = self.wound_stage_prev
            else:
                # use previous prob
                probs = np.array([(float(df_im.iat[-1, 4 + pidx])) for pidx in range(4)])
        else:
            # reset inbetween time for flx delivery within 2 hour window
            self.drug_inbetween_total = 0.0
            self.new_wound_stage_flag = True

        self.imwriter.probs_plot(probs, self.cnters, '{}_Prediction'.format(estimator), self.wound_num)
        self.cnters['{}_Prediction'.format(estimator)] += 1
        self.wound_stage_prev = copy.deepcopy(probs)
        return probs, m_size_im

    def switchType(self, probs):
        if probs[1] >= 0.3:
                self.treatmentEF = False

    def closed_loop(self, new_state):
        '''
        Closed loop control using deep reinforcement learning
        @return:
        '''

        self.state_next = copy.deepcopy(new_state)
        # self.state_next = self.mapper.ws_est(dirtmp)
        reward = -1 if self.state_next[-1] < 0.95 else 0
        done = False if self.state_next[-1] < 0.95 else True
        # action = self.act_space[self.rlagent.act(self.state_cur, eps=self.eps)]
        actionIdx = self.rlagent.act(self.state_cur, eps=self.eps)
        actuation = self.act_space[actionIdx]
        # if self.state_next[0] >= 0.9 or self.state_next[0] <= 0.1 or self.read_gui():
        #     actuation = self.zero_act
        if self.state_next[0] <= 0.1 or self.read_gui():
            actuation = self.zero_act
        self.rlagent.step(self.state_cur, actionIdx, reward, self.state_next, done)
        # update next state
        self.state_cur = copy.deepcopy(self.state_next)

        self.ref_prev = self.ref
        self.ref = actuation
        self.eps = max(self.rlagent.args.eps_end, self.rlagent.args.eps_decay * self.eps)  # decrease epsilon
        # print('Wound Prob: {} \t Actuation: {:.4f}'.format(self.state_cur, self.ref))
        self.rlcnt += 1

    def control(self):
        # first, we need to set some hyper-parameters from outside command
        # TODO: let the party begin
        # current healnet prob table size

        if not os.path.exists(self.deviceArgs.deepmapper_prob_file_name):
            m_size_old_im_deepmapper = 0
        else:
            m_size_old_im_deepmapper = len(self.read_csv(self.deviceArgs.deepmapper_prob_file_name).index)

        # feed back from device
        start_time = time.time()

        while True:
            if (time.time() - start_time) > (self.hours * 3600 + self.additional_time):
                self.ref_prev, self.ref = 0.0, 0.0
                self.save_csv_control()
                print("The code has been run {:.4f} hours".format((time.time() - start_time) / 3600))
                break

            # close loop control
            self.pred_deepmapper()
            deepmapper_probs, m_size_old_im_deepmapper = self._wst(m_size_old_im_deepmapper, "DeepMapper")

            if self.deviceArgs.treatment == 2:
                self.drug_const()
            if self.new_wound_stage_flag and (not self.shut_off_flag) and (not self.drug_max_flag):
                self.closed_loop(deepmapper_probs)
            print('Wound Prob: {} 	\t Actuation: {:.4f} Drug: {:.4f} mg/mol'.format(deepmapper_probs, self.ref, self.drug_total))

            # constantly check gui output
            if not self.shut_off_flag:
                self.read_gui()
            # save control info to csv
            if self.new_wound_stage_flag:
                # print("New wound images...")
                self.save_csv_control()
                self.new_wound_stage_flag = False

            # we read data every 3 seconds, plot into tensorboard, and save data for physicians GUI
            try:
                self.plot()
            except:
                print('Failed to combine and plot all data, try next time')
            time.sleep(2.5)
            self.cnters['control_step'] += 1

    def nocotrl(self):
        # first, we need to set some hyper-parameters from outside command
        # TODO: let the party begin
        # current healnet prob table size

        if not os.path.exists(self.deviceArgs.deepmapper_prob_file_name):
            m_size_old_im_deepmapper = 0
        else:
            m_size_old_im_deepmapper = len(self.read_csv(self.deviceArgs.deepmapper_prob_file_name).index)

        # feed back from device
        # timers time
        start_time = time.time()

        while True:
            if (time.time() - start_time) > (self.hours * 3600 + self.additional_time):
                self.ref_prev, self.ref = 0.0, 0.0
                self.save_csv_control()
                print("The code has been run {:.4f} hours".format((time.time() - start_time) / 3600))
                break

            self.pred_deepmapper()
            deepmapper_probs, m_size_old_im_deepmapper = self._wst(m_size_old_im_deepmapper, "DeepMapper")
            print('Wound Prob: {}'.format(deepmapper_probs))
            # self.open_loop(healnet_probs)
            # constantly check guix output
            if not self.shut_off_flag:
                self.read_gui()
            # save control info to csv
            if self.new_wound_stage_flag:
                # print("New wound images...")
                # self.save_csv_control()
                self.new_wound_stage_flag = False

            # we read data every 3 seconds, plot into tensorboard, and save data for physicians GUI
            try:
                self.plot()
            except:
                print('Failed to combine and plot all data, try next time')
            time.sleep(2.5)
            self.cnters['control_step'] += 1

    def comControlOpt12(self):
        # first, we need to set some hyper-parameters from outside command
        # TODO: let the party begin
        # current healnet prob table size

        if not os.path.exists(self.deviceArgs.deepmapper_prob_file_name):
            m_size_old_im_deepmapper = 0
        else:
            m_size_old_im_deepmapper = len(self.read_csv(self.deviceArgs.deepmapper_prob_file_name).index)

        # feed back from device
        # timers time
        start_time = time.time()

        while True:
            if (time.time() - start_time) > (self.hours * 3600 + self.additional_time):
                self.ref_prev, self.ref = 0.0, 0.0
                self.save_csv_control()
                print("The code has been run {:.4f} hours".format((time.time() - start_time) / 3600))
                break

            # close loop control
            self.pred_deepmapper()
            deepmapper_probs, m_size_old_im_deepmapper = self._wst(m_size_old_im_deepmapper, "DeepMapper")

            self.switchType(deepmapper_probs)

            if self.treatmentEF:
                # EF
                if self.new_wound_stage_flag and (not self.shut_off_flag) and (not self.drug_max_flag) and (not self.drug_inbetween_flag):
                    self.closed_loop(deepmapper_probs)
            else:
                # Flx
                # change Action Space to Flx
                self.act_space = copy.deepcopy(self.act_Flx_space)
                self.drug_const()
                if self.new_wound_stage_flag and (not self.shut_off_flag) and (not self.drug_max_flag) and (not self.drug_inbetween_flag):
                    self.closed_loop(deepmapper_probs)

            treats = {1: 'EF', 0: 'Flx'}
            print('Treat {} Dur: {:.2f}min \t Wound Prob: [H: {:.2f} I: {:.2f} P: {:.2f} M: {:.2f}] \t TC: {:.2f}uA Flx: {:.4f} mg/mol'.format(
                treats[int(self.treatmentEF)], (time.time() - start_time)/60,
                deepmapper_probs[0], deepmapper_probs[1], deepmapper_probs[2], deepmapper_probs[3],
                self.ref, self.drug_total))

            # constantly check gui output
            if not self.shut_off_flag:
                self.read_gui()
            # save control info to csv
            if self.new_wound_stage_flag:
                # print("New wound images...")
                self.save_csv_control()
                self.new_wound_stage_flag = False

            # we read data every 3 seconds, plot into tensorboard, and save data for physicians GUI
            try:
                self.plot()
            except:
                print('Failed to combine and plot all data, try next time')
            time.sleep(2.5)
            self.cnters['control_step'] += 1


if __name__ == "__main__":
    xrange = np.linspace(0, 20, 240)
    y_opt = odeint(simple, np.array([1., 0., 0., 0.]), xrange, args=(np.array([0.5, 0.3, 0.1]),))
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state=0)
    # fit the regressor with X and Y data
    regressor.fit(y_opt, xrange)
    print(regressor.predict(np.array([0.0, 0.0, 0.1, 0.9]).reshape(1, -1)))