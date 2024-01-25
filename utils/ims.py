

import numpy as np
import copy
import cv2
import pywt
import os
import time
from PIL import Image

import torch.multiprocessing as mp


def merge_zstack(directory):
    #     print('{} Begin!!!'.format(directory))
    # Get a list of the image files
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')])

    # Read the images
    images = [cv2.imread(os.path.join(directory, img)) for img in image_files]

    # Initialize final_coeffs for each channel
    first_wt = pywt.wavedec2(cv2.split(images[0])[0], 'db1', level=2)

    # Create deep copies of the initial coefficients for each channel
    final_coeffs_R = copy.deepcopy(first_wt)
    final_coeffs_G = copy.deepcopy(first_wt)
    final_coeffs_B = copy.deepcopy(first_wt)

    # Loop over each channel
    for channel in range(3):
        wavelet_transforms = [pywt.wavedec2(cv2.split(img)[channel], 'db1', level=2) for img in images]
        for i in range(len(wavelet_transforms[0])):
            for j in range(len(wavelet_transforms[0][i])):
                max_coeff = np.max([wt[i][j] for wt in wavelet_transforms], axis=0)
                if channel == 0:
                    final_coeffs_R[i][j][:] = max_coeff
                elif channel == 1:
                    final_coeffs_G[i][j][:] = max_coeff
                else:
                    final_coeffs_B[i][j][:] = max_coeff

    # Compute inverse wavelet transform for each channel
    final_image_R = pywt.waverec2(final_coeffs_R, 'db1')
    final_image_G = pywt.waverec2(final_coeffs_G, 'db1')
    final_image_B = pywt.waverec2(final_coeffs_B, 'db1')

    # Make sure they are in uint8 format and in the range [0, 255]
    final_image_R = np.clip(final_image_R, 0, 255).astype('uint8')
    final_image_G = np.clip(final_image_G, 0, 255).astype('uint8')
    final_image_B = np.clip(final_image_B, 0, 255).astype('uint8')

    # Merge channels back but in RGB order
    final_image = cv2.merge((final_image_B, final_image_G, final_image_R))  # Changed the order here
    final_image = Image.fromarray(final_image, 'RGB')

    # save image to the corresponding folder
    # im = Image.fromarray(final_image, 'RGB')
    # im.save(directory + '/merged_{}.jpg'.format(directory.split('/')[-1]))

    return final_image, final_coeffs_R, final_coeffs_G, final_coeffs_B


if __name__ == "__main__":
    process = []

    for im in range(7):
        for i in range(7):
            time.sleep(0.1)
            p = mp.Process(target=merge_zstack, args=(dir_tmpss[7 * im + i],))
            p.start()
            process.append(p)

        for p in process:
            p.join()
            time.sleep(0.1)