import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import gin

def _get_dataframe():
    raw_datadir = gin.query_parameter('make_tfrecords.data_dir')
    file_list = os.listdir(raw_datadir)
    file_list.sort()
    acc_data_list = []
    gyr_data_list = []

    for filename in file_list:
        file_path = raw_datadir+filename
        if 'acc' in filename:
            acc_data_list.append(pd.read_csv(file_path, delimiter=' ', header=None, names=['acc_x', 'acc_y', 'acc_z']))
        if 'gyro' in filename:
            gyr_data_list.append(pd.read_csv(file_path, delimiter=' ', header=None, names=['gyro_x', 'gyro_y', 'gyro_z']))
        if 'label' in filename:
            label_data = pd.read_csv(file_path, delimiter=' ', header=None, names=['exp_id', 'user_id', 'activity_id', 'start_point', 'end_point'])

    df_full_list=[]
    for experiment in range(label_data['exp_id'].max()):
        labels_data_expt = label_data[label_data['exp_id'] == experiment+1] #because the experiment 1 has list index 0
        user_id = labels_data_expt.iloc[0]['user_id']
        df_full_list.append(pd.concat([gyr_data_list[experiment], acc_data_list[experiment]], axis=1, ignore_index=False))
        df_full_list[-1]['user_id'] = user_id
        df_full_list[-1]['activity_id'] = -1
        df_full_list[-1]['exp_id'] = experiment + 1

        for i, row in labels_data_expt.iterrows():
            start = row['start_point']
            end = row['end_point']
            activity_id = row['activity_id']
            df_full_list[experiment].loc[(df_full_list[experiment].index >= start) & (df_full_list[experiment].index <= end), 'activity_id'] = activity_id
    
    return df_full_list

def visualise_HAR(experiment_number):
    df_list = _get_dataframe()
    df = df_list[experiment_number-1]
    categories = np.unique(df['activity_id'])
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    colordict = dict(zip(categories, colors))

    axes = ['x', 'y', 'z']
    sensor_types = ['acc', 'gyro']
    activity_labels = ['No activity', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING',
                    'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE',
                    'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']

    fig, axs = plt.subplots(len(axes), len(sensor_types), figsize=(15, 5), sharex=True)

    for k, sensor in enumerate(sensor_types):
        for j, axis in enumerate(axes):
            segments = []
            colors = []

            for i in range(len(df) - 1):
                x = [df.index[i], df.index[i + 1]]
                y = [df[f'{sensor}_{axis}'].iloc[i], df[f'{sensor}_{axis}'].iloc[i + 1]]
                category_color = colordict[df['activity_id'].iloc[i]]
                segments.append(list(zip(x, y)))
                colors.append(category_color)

            lc = LineCollection(segments, colors=colors, cmap='tab20', norm=Normalize(0, len(categories) - 1))
            axs[j, k].add_collection(lc)
            axs[j, k].autoscale()
            axs[j, k].set_ylim(df[f'{sensor}_{axis}'].min(), df[f'{sensor}_{axis}'].max())
            axs[j, k].set_ylabel(f'{sensor}_{axis}')

    # Add colorbar
    cbar = plt.colorbar(lc, ax=axs.ravel().tolist(), ticks=np.arange(len(categories)), label='Activity ID')
    cbar.set_label('Activity labels')
    cbar.set_ticklabels(activity_labels)

    plt.savefig('visualise_har.png')
