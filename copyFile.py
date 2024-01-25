
import time
import os
import shutil

dsk_dir = os.path.expanduser("~/Desktop/")
org_dir = dsk_dir + 'imtmps/Wound_6/'
dst_dir = dsk_dir + 'Porcine_Exp_Davis/Wound_1/'

all_ims = sorted(os.listdir(org_dir))

for tmp in all_ims:
    if tmp.startswith('20'):
        print(tmp, org_dir, dst_dir)
        if not os.path.exists(dst_dir + tmp):
            shutil.copytree(org_dir + tmp, dst_dir + tmp)
            time.sleep(600)