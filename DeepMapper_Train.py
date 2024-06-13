import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import time

from algs.deepmapper import *

from cfgs.config_device import DeviceParameters
from torch.utils.tensorboard import SummaryWriter
from algs.alphaHeal import AlphaHeal


# pathlib: Object-oriented filesystem paths
from pathlib import Path

import torch.multiprocessing as mp

from utils.tools import *

def merge_one_process(dir_exp_cam_date, gen_dir, dir_exp, dir_cam):

    for dir_date in os.listdir(dir_exp_cam_date):

        dir_exp_cam_data_ims = dir_exp_cam_date + dir_date

        gen_dirs = gen_dir + dir_exp + '/' + dir_cam + '/'
        if not os.path.exists(gen_dirs):
            os.makedirs(gen_dirs)
        img_dir_com = dir_exp_cam_data_ims
        # merging and downsampling
        # print('Merging generated exp {} cam {} images of {}...\n'.format(dir_exp, dir_cam, dir_date))
        t1 = time.time()
        im_name = img_dir_com.__str__().split('/')[-1]
        im_name = gen_dirs + im_name + '.png'

        if not os.path.exists(im_name):
            final_image, final_coeffs_R, final_coeffs_G, final_coeffs_B = merge_zstack(img_dir_com)
            final_image = final_image.resize((128, 128))
            final_image.save(im_name)
        t2 = time.time()
        print('Time Taken: {:.4f} \n Finish image generation exp {} cam {} images of {}. \t'.format( t2 - t1, dir_exp, dir_cam, dir_date))


def im_gen():
    main_dir = 'E:/Data/Porcine_Exp_Davis/'
    gen_dir = 'E:/Data/Porcine_Exp_Davis_Processed/'

    for dir_exp in os.listdir(main_dir):
        dir_exp_cam = main_dir + dir_exp + '/'

        all_cam_dirs = os.listdir(dir_exp_cam)
        idx = 0
        while True:
            if idx >= len(all_cam_dirs):
                break
            process = []
            for nn in range(4):
                if idx >= len(all_cam_dirs):
                    break
                dir_cam = all_cam_dirs[idx]
                dir_exp_cam_date = dir_exp_cam + dir_cam + '/'
                time.sleep(0.1)
                p = mp.Process(target=merge_one_process, args=(dir_exp_cam_date, gen_dir, dir_exp, dir_cam,))
                p.start()
                process.append(p)
                idx += 1
            for p in process:
                p.join()
                time.sleep(0.1)


def one_trajectory(look_ahead_cnt, image_paths, mapper):
    avg_loss, cnt = 0.0, 0.0

    for idx in range(len(image_paths) - look_ahead_cnt):

        curr_device_image = mapper.process_im(image_paths[idx])
        next_device_image = mapper.process_im(image_paths[idx + 1])
        next4_device_image = mapper.process_im(image_paths[idx + look_ahead_cnt])

        curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
        curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(mapper.device)

        next_image_data = np.expand_dims(next_device_image.T, axis=0)
        next_image_data = torch.from_numpy(next_image_data / 255.0).float().to(mapper.device)

        next4_image_data = np.expand_dims(next4_device_image.T, axis=0)
        next4_image_data = torch.from_numpy(next4_image_data / 255.0).float().to(mapper.device)

        prob, A_prob, x_hat, x_next_hat = mapper.model(curr_image_data)
        prob_next, _, _, _ = mapper.model(next_image_data)
        prob_next4, _, _, _ = mapper.model(next4_image_data)
        mapper.optimizer.zero_grad()

        A_prob_shift = mapper.model.shift(A_prob)
        for xxx in range(look_ahead_cnt - 1):
            A_prob_shift = mapper.model.shift(A_prob_shift)

        if idx <= 0:
            loss = 0.2 * mapper.criterion(x_hat, curr_image_data) + \
                   0.2 * mapper.criterion(x_next_hat, next_image_data) + \
                   0.4 * mapper.criterion(prob, mapper.init_prob) + \
                   0.05 * mapper.criterion(A_prob, prob_next.detach()) + \
                   0.05 * mapper.criterion(prob_next, A_prob.detach()) + \
                   0.1 * mapper.criterion(A_prob_shift, prob_next4)
        else:
            loss = 0.2 * mapper.criterion(x_hat, curr_image_data) + \
                   0.2 * mapper.criterion(x_next_hat, next_image_data) + \
                   0.15 * mapper.criterion(A_prob, prob_next.detach()) + \
                   0.15 * mapper.criterion(prob_next, A_prob.detach()) + \
                   0.3 * mapper.criterion(A_prob_shift, prob_next4)

        loss.backward()
        mapper.optimizer.step()

        avg_loss += loss.cpu().item()
        cnt += 1


        return avg_loss / cnt, mapper


def train():
    desktop_dir = os.path.expanduser("~/Desktop/")
    deviceArgs = DeviceParameters(9, desktop_dir)

    deviceArgs.expNum = 99
    deviceArgs.isTrain = True

    alg_dir = desktop_dir + 'Close_Loop_Actuation/'
    alg_out = alg_dir + 'Output/'
    alg_res = alg_dir + 'data_save/'

    runs_device = '_'.join(('_'.join(time.asctime().split(' '))).split(':'))
    runs_device = alg_dir + 'runs/deepmapper/{}'.format(runs_device)
    deviceArgs.runs_device = runs_device
    dirs = [alg_out, alg_res, runs_device]

    writer = SummaryWriter(log_dir=deviceArgs.runs_device)
    mapper = DeepMapper(deviceArgs=deviceArgs, writer=writer)

    look_ahead_cnt = 4
    main_dir = 'E:/data/Porcine_Exp_Davis_Processed/'

    mapper.num_epochs = int(5e4)

    final_prob_ref = torch.from_numpy(np.array([0, 0, 0, 1.0])).view(1, -1).float().to(mapper.device)

    for ep in range(mapper.num_epochs):
        # dir_exp_cam_date = main_dir + 'exp_10/' + 'Wound_6/'
        # root_images = Path(dir_exp_cam_date)
        # image_paths = list(root_images.glob("*.png"))

        avg_loss, cnt = 0.0, 0.0
        for dir_exp in os.listdir(main_dir):
            dir_exp_cam = main_dir + dir_exp + '/'
            # print('Training with {} images \t ep: {}/{}'.format(dir_exp_cam, ep, mapper.num_epochs))
            for dir_cam in os.listdir(dir_exp_cam):
                dir_exp_cam_date = dir_exp_cam + dir_cam + '/'
                root_images = Path(dir_exp_cam_date)
                image_paths = list(root_images.glob("*.png"))
                # avg_loss = one_trajectory(look_ahead_cnt, image_paths, mapper)

                time_process = str2sec(str(image_paths[0])[-20:-5]) - 7200.0
                for idx in range(len(image_paths) - look_ahead_cnt):

                    curr_device_image = mapper.process_im(image_paths[idx])
                    next_device_image = mapper.process_im(image_paths[idx + 1])
                    next4_device_image = mapper.process_im(image_paths[idx + look_ahead_cnt])

                    curr_image_data = np.expand_dims(curr_device_image.T, axis=0)
                    curr_image_data = torch.from_numpy(curr_image_data / 255.0).float().to(mapper.device)

                    next_image_data = np.expand_dims(next_device_image.T, axis=0)
                    next_image_data = torch.from_numpy(next_image_data / 255.0).float().to(mapper.device)

                    next4_image_data = np.expand_dims(next4_device_image.T, axis=0)
                    next4_image_data = torch.from_numpy(next4_image_data / 255.0).float().to(mapper.device)

                    time_dif1 = str2sec(str(image_paths[idx])[-20:-5]) - time_process
                    time_dif2 = str2sec(str(image_paths[idx + 1])[-20:-5]) - str2sec(str(image_paths[idx])[-20:-5])
                    time_dif4 = str2sec(str(image_paths[idx + look_ahead_cnt])[-20:-5]) - str2sec(str(image_paths[idx + look_ahead_cnt - 1])[-20:-5])
                    time_process = str2sec(str(image_paths[idx])[-20:-5])

                    prob, A_prob, x_hat, x_next_hat = mapper.model(curr_image_data, time_dif1)
                    prob_next, _, _, _ = mapper.model(next_image_data, time_dif2)
                    prob_next4, _, _, _ = mapper.model(next4_image_data, time_dif4)
                    mapper.optimizer.zero_grad()

                    A_prob_shift = mapper.model.shift(A_prob, time_dif2)
                    for xxx in range(look_ahead_cnt - 1):
                        time_dif = str2sec(str(image_paths[idx + xxx + 1])[-20:-5]) - str2sec(str(image_paths[idx + xxx])[-20:-5])
                        A_prob_shift = mapper.model.shift(A_prob_shift, time_dif)

                    if idx <= 0:
                        loss = 0.2 * mapper.criterion(x_hat, curr_image_data) + \
                               0.2 * mapper.criterion(x_next_hat, next_image_data) + \
                               0.4 * mapper.criterion(prob, mapper.init_prob) + \
                               0.05 * mapper.criterion(A_prob, prob_next.detach()) + \
                               0.05 * mapper.criterion(prob_next, A_prob.detach()) + \
                               0.1 * mapper.criterion(A_prob_shift, prob_next4)
                    elif idx < len(image_paths) - look_ahead_cnt - 1:
                        loss = 0.2 * mapper.criterion(x_hat, curr_image_data) + \
                               0.2 * mapper.criterion(x_next_hat, next_image_data) + \
                               0.15 * mapper.criterion(A_prob, prob_next.detach()) + \
                               0.15 * mapper.criterion(prob_next, A_prob.detach()) + \
                               0.3 * mapper.criterion(A_prob_shift, prob_next4)
                    else:
                        A_prob_shift_end = A_prob_shift
                        for xxx in range(180):
                            A_prob_shift_end = mapper.model.shift(A_prob_shift_end, 7200)

                        loss = 0.2 * mapper.criterion(x_hat, curr_image_data) + \
                               0.2 * mapper.criterion(x_next_hat, next_image_data) + \
                               0.15 * mapper.criterion(A_prob, prob_next.detach()) + \
                               0.15 * mapper.criterion(prob_next, A_prob.detach()) + \
                               0.15 * mapper.criterion(A_prob_shift, prob_next4) + \
                               0.15 * mapper.criterion(A_prob_shift_end, final_prob_ref)

                    loss.backward()
                    mapper.optimizer.step()

                    avg_loss += loss.cpu().item()
                    cnt += 1

        if (ep + 1) % 50 == 0:
            print('Train Epoch [{}/{}] Loss:{:.4f}'.format(ep + 1, mapper.num_epochs, avg_loss))

            dir_exp_cam_date = 'E:/Data/Porcine_Exp_Davis_Processed/exp_14/Camera_B/'
            root_images = Path(dir_exp_cam_date)
            image_paths = list(root_images.glob("*.png"))

            mapper.test(ep, image_paths)

            torch.save(mapper.model.state_dict(), mapper.args.model_dir + 'checkpoint_ep_{}.pth'.format(ep))
            mapper.writer.add_scalar('Loss/train_mse', avg_loss, ep)
            mapper.writer.add_scalar('Ks/k_h', mapper.model.Kh, ep)
            mapper.writer.add_scalar('Ks/k_i', mapper.model.Ki, ep)
            mapper.writer.add_scalar('Ks/k_p', mapper.model.Kp, ep)


def ds_merge(im_dir, imdata_dir):
    '''
    dwonsampling merging
    @im_dir: image directory where device image stored
    @return:
    '''

    im_name = im_dir.__str__().split('/')[-1]
    im_name = imdata_dir + im_name + '.png'


    if not os.path.exists(im_name):
        final_image, final_coeffs_R, final_coeffs_G, final_coeffs_B = merge_zstack(im_dir)
        final_image = final_image.resize((128, 128))
        final_image.save(im_name)
    return im_name


def test(wnum, expNum):

    desktop_dir = os.path.expanduser("~/Desktop/")
    deviceArgs = DeviceParameters(wnum, desktop_dir)

    deviceArgs.expNum = expNum
    deviceArgs.isTrain = False

    alg_dir = desktop_dir + 'Close_Loop_Actuation/'
    alg_out = alg_dir + 'Output/'
    alg_res = alg_dir + 'data_save/'

    runs_device = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + 'wound_{}'.format(deviceArgs.wound_num)
    runs_device = alg_dir + 'runs/deepmapper/{}'.format(runs_device)
    deviceArgs.runs_device = runs_device
    dirs = [alg_out, alg_res, runs_device]


    wound = AlphaHeal(deviceArgs=deviceArgs)

    # writer = SummaryWriter(log_dir=deviceArgs.runs_device)
    # mapper = DeepMapper(deviceArgs=deviceArgs, writer=writer)
    #
    # mapper.model.load_state_dict(torch.load(desktop_dir + 'Close_Loop_Actuation/models/deepmapper_ep_final.pth'))

    dir_exp_cam_date = desktop_dir + 'Close_Loop_Actuation/data_save/exp_{}/deepmapper/data_wound_{}/dsmgIMS/'.format(expNum, wnum)
    root_images = Path(dir_exp_cam_date)
    image_paths = list(root_images.glob("*.png"))

    wound.mapper.test(0, image_paths, wound.progressor.predict)


def MergeImsSignleP(wnum):

    maptable = {1: 'A', 2: 'B', 5: 'D', 6: 'E'}
    desktop_dir = os.path.expanduser("~/Desktop/")

    im_dir = desktop_dir + 'imtmps/Exp_22/Camera_{}/'.format(maptable[wnum])
    dir_exp_cam_date = desktop_dir + 'Close_Loop_Actuation/data_save/exp_22/deepmapper/data_wound_{}/dsmgIMS/'.format(wnum)
    if not os.path.exists(dir_exp_cam_date):
        os.makedirs(dir_exp_cam_date)

    for dir_tmp in os.listdir(im_dir):
        try:
            print("Start Merging {} \n".format(dir_tmp))
            t1 = time.time()
            im_name = ds_merge(im_dir + dir_tmp, dir_exp_cam_date)
            t2 = time.time()
            print("Finish Merging!!! Time: {:.2f} \n".format(t2 - t1))
        except:
            print("No Images found in {} \n".format(dir_tmp))

def MergeImsMultiP():

    processes  = []
    maptable = {1: 'A', 2: 'B', 5: 'D', 6: 'E'}
    for wnum in maptable.keys():
        p = mp.Process(target=MergeImsSignleP, args=(wnum, ))
        processes.append(p)
        p.start()

    for p in processes :
        p.join()

if __name__ == "__main__":
    train()
    # test(wnum=3, expNum=14)
    # im_gen()