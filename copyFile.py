
import time
import os
import shutil

def copy1(org_dir, dst_dir, delay):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    all_ims = sorted(os.listdir(org_dir))

    for tmp in all_ims:
        if tmp.startswith('20'):
            print(tmp, org_dir, '\n', dst_dir)
            if not os.path.exists(dst_dir + tmp):
                shutil.copytree(org_dir + tmp, dst_dir + tmp)
                time.sleep(delay)

if __name__ == "__main__":

    delay = input('Delay Time in Seconds: ')
    delay = int(delay)

    orgcam = input('Org Camera Label: ')
    orgcam = str(orgcam)

    dstcam = input('Dst Camera Label: ')
    dstcam = str(dstcam)

    orgexn = input('Org Exp Num: ')
    orgexn = str(orgexn)

    dstexn = input('Dst Exp Num: ')
    dstexn = str(dstexn)

    dsk_dir = os.path.expanduser("~/Desktop/")
    org_dir = dsk_dir + 'imtmps/Exp_{}/Camera_{}/'.format(orgexn, orgcam)
    dst_dir = dsk_dir + 'Porcine_Exp_Davis/Exp_{}/Camera_{}/'.format(dstexn, dstcam)
    # os.rmdir(dsk_dir)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    copy1(org_dir, dst_dir, delay)