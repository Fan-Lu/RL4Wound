####################################################
# Description: Test Function of RL4Wound
# Version: V0.1.1
# Author: Fan Lu @ UCSC
# Data: 2023-12-29
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
import time

import matplotlib.pyplot as plt
import pandas as pd

from algs.alphaHeal import AlphaHeal
from cfgs.config_device import DeviceParameters

from algs.deepmapper import *

# pathlib: Object-oriented filesystem paths
from pathlib import Path

from utils.tools import *

def deepmapper(data_dir, image_dir, deviceArgs, wound_num, expnum, ep):
    writer = SummaryWriter(log_dir=deviceArgs.runs_device)
    mapper = DeepMapper(deviceArgs, writer)

    mapper.model.load_state_dict(torch.load(deviceArgs.desktop_dir + 'Close_Loop_Actuation/data_save/exp_99/deepmapper/models_wound_9/checkpoint_ep_{}.pth'.format(ep)))

    im_gens = []
    im_orgs = []

    xrange = np.linspace(0, 3, len(image_dir) + 1)
    prob_buf = np.array([1., 0., 0., 0.])
    time_process = str2sec(image_dir[0][-20:-5]) - 7200.0
    for idx in range(len(image_dir)):
        # xrange.append((xrange[-1] + 1) * 2)
        curr_device_image = mapper.process_im(image_dir[idx])
        curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
        curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(mapper.device)

        time_dif = str2sec(image_dir[idx][-20:-5]) - time_process
        time_process = str2sec(image_dir[idx][-20:-5])
        prob, A_prob, x_hat, x_next_hat = mapper.model(curr_image_data, time_dif)
        prob_buf = np.vstack((prob_buf, A_prob.cpu().data.numpy().squeeze()))

        x_hat_np = x_hat.data.numpy().squeeze().T
        x_hat_np = x_hat_np * 255
        im_hat = Image.fromarray(x_hat_np.astype(np.uint8))
        im_org = Image.fromarray((curr_image_data.data.numpy().squeeze().T * 255).astype(np.uint8))

        im_gens.append(im_hat)
        im_orgs.append(im_org)

    numIms = len(im_orgs)
    numRows = numIms // 7
    if numIms % 7 > 0:
        numRows += 1

    dst1 = Image.new('RGB', (128 * 7, 128 * numRows))
    dst2 = Image.new('RGB', (128 * 7, 128 * numRows))
    for j in range(numRows):
        for i in range(7):
            if (i + j * 7) < numIms:
                dst1.paste(im_orgs[i + j * 7], (i * 128, 128 * j))
                dst2.paste(im_gens[i + j * 7], (i * 128, 128 * j))

    xrangeAll = np.linspace(0, 20, 240)

    # TODO: Fix Xrange
    y_tmp = odeint(simple, np.array([1., 0., 0., 0.]), xrangeAll,
                   args=(np.array([F.sigmoid(mapper.model.Kh).data.cpu().numpy()[0],
                                   F.sigmoid(mapper.model.Ki).data.cpu().numpy()[0],
                                   F.sigmoid(mapper.model.Kp).data.cpu().numpy()[0]]),))

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax.scatter(xrange, prob_buf[:, 0], color='r', label='Hemostasis')
    ax.scatter(xrange, prob_buf[:, 1], color='g', label='Inflammation')
    ax.scatter(xrange, prob_buf[:, 2], color='b', label='Proliferation')
    ax.scatter(xrange, prob_buf[:, 3], color='y', label='Maturation')

    # ax.plot(xrangeAll, y_tmp[:, 0], color='r', label='H')
    # ax.plot(xrangeAll, y_tmp[:, 1], color='g', label='I')
    # ax.plot(xrangeAll, y_tmp[:, 2], color='b', label='P')
    # ax.plot(xrangeAll, y_tmp[:, 3], color='y', label='M')

    ax.set_title('Wound_{}'.format(wound_num))
    # leg_pos = (1, 0.5)
    # ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax.set_xlabel('Time (day)')
    plt.tight_layout()

    plt.savefig('{}probs_exp_{}_wound_{}.png'.format(data_dir, expnum, wound_num))
    dst1.save('{}org_exp_{}_wound_{}.png'.format(data_dir, expnum, wound_num))
    dst2.save('{}gen_exp_{}_wound_{}.png'.format(data_dir, expnum, wound_num))

    plt.close()
    dst1.close()
    dst2.close()


if __name__ == '__main__':
    expnum = 21
    ep = 9999

    map_table = pd.read_csv('C:/Users/fanlu/Desktop/Close_Loop_Actuation/data_save/mapping_tables/Device-to-Wound-Mapping-Table_exp_{}.csv'.format(expnum))
    data_dir = 'C:/Users/fanlu/Desktop/Close_Loop_Actuation/data_save/exp_{}_post/'.format(expnum)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    for i in range(len(map_table)):
        wound_num, wound_abc = map_table[['Wound No.', 'Camera Identifier']].values[i]
        desktop_dir = os.path.expanduser("~/Desktop/")
        deviceArgs = DeviceParameters(wound_num, desktop_dir)
        image_dir = 'D:/WoundDataDARPA/Porcine_Exp_Davis_Processed_all/exp_{}/{}/'.format(expnum, wound_abc)
        image_dirs = []
        for tmp in os.listdir(image_dir):
            image_dirs.append(image_dir + tmp)

        deepmapper(data_dir, image_dirs, deviceArgs, wound_num, expnum, ep)



