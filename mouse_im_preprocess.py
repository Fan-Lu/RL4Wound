
import os
import numpy as np
import pandas as pd


def create_csv(train=True):
    if train:
        imdir = '../../../../WoundDataDARPA/MouseData/train/'
    else:
        imdir = '../../../../WoundDataDARPA/MouseData/val/'

    df = {
        'ImNa': [],
        'Day': [],
        'Age': [],
        'WNum': [],
        'Side': [],
    }

    ndays = len(os.listdir(imdir))

    for day in range(ndays):
        ims_dir_tmp = imdir + '{}/'.format(day)
        for im in os.listdir(ims_dir_tmp):
            _, info = im.split(' ')
            (d, acst) = info.split('_')
            (a, c, st) = acst.split('-')
            s, _ = st.split('.')
            df['ImNa'].append(im)
            df['Day'].append(d)
            df['Age'].append(a)
            df['WNum'].append(c)
            df['Side'].append(s)

    dataframe = pd.DataFrame(df)
    if train:
        dataframe.to_csv(imdir + '../all_train_imgs.csv')
    else:
        dataframe.to_csv(imdir + '../all_test_imgs.csv')

    if train:
        adualt_wnums = set(dataframe.loc[(dataframe.Age == 'A8')].WNum.values)
        young_wnums = set(dataframe.loc[(dataframe.Age == 'Y8')].WNum.values)
        sides = ['L', 'R']

        for an in adualt_wnums:
            for s in sides:
                df_tmp = dataframe.loc[(dataframe.Age == 'A8') & (dataframe.WNum == an) & (dataframe.Side == s)]
                df_tmp.to_csv(imdir + '../all_train_imgs_A8_{}_{}.csv'.format(an, s))

        for an in young_wnums:
            for s in sides:
                df_tmp = dataframe.loc[(dataframe.Age == 'A8') & (dataframe.WNum == an) & (dataframe.Side == s)]
                df_tmp.to_csv(imdir + '../all_train_imgs_Y8_{}_{}.csv'.format(an, s))


def create_imdirs_from_csv(age, cnt, side):
    imdir = '../../../../WoundDataDARPA/MouseData/'

    im_paths = []
    df = pd.read_csv(imdir + 'all_train_imgs.csv')
    df_tmp = df.loc[(df.Age == age) & (df.WNum == cnt) & (df.Side == side)]

    for i in range(len(df_tmp)):
        dir_tmp = imdir + 'train/' + '{}/'.format(df_tmp.Day.iloc[i]) + df_tmp.ImNa.iloc[i]
        im_paths.append(dir_tmp)

    return im_paths


if __name__ == "__main__":

    create_csv(False)
    # create_imdirs_from_csv('A8', 3, 'L')