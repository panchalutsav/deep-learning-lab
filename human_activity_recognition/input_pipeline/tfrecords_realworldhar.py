import gin, logging, os, sys
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from scipy.stats import zscore, mode
import numpy as np
from sklearn.utils import shuffle
import re
import matplotlib.pyplot as plt
import sys


labels = {
    'climbingdown': 1,
    'climbingup': 2,
    'jumping': 3,
    'lying': 4,
    'standing': 5,
    'sitting': 6,
    'running': 7,
    'walking': 8,
}

# tf.window() does not create exact window_size samples, so we need to define out own custom window maker. 
# below is another function that makes the use of tf.window() function
def custom_window_maker(data, window_size, shift):
  features=[]
  labels=[]
  for i in range(0, int(data.shape[0]/shift) -1):
    start = i*shift
    end = i*shift + window_size
    one_window = data[start:end].values
    one_window_features = one_window[:, :-1]
    one_window_labels = one_window[:, -1]
    label = mode(one_window_labels, keepdims=False).mode
    features.append(one_window_features)
    labels.append(label)
  features = np.array(features)
  labels = np.array(labels)
  labels = np.expand_dims(labels, axis=1)
  return (features, labels)
  

# --- deprecated
def window_maker(data, is_train_data, window_size, shift, low_limit=0):
  features_list = []
  labels_list = []
  tf_dataset_normalized = tf.data.Dataset.from_tensor_slices(data)  # TODO: make label changes
  windowed_data = tf_dataset_normalized.window(size=window_size, shift=shift, drop_remainder=True)
  flat_windowed_dataset = windowed_data.flat_map(lambda window: window.batch(window_size))
  windows_as_arrays = list(flat_windowed_dataset.as_numpy_iterator())
  for window in windows_as_arrays:
    features = window[:, :-1]
    label, count = mode(window[:, -1], keepdims=False).mode, mode(window[:, -1]).count # setting keepdims to False will prevent adding extra axis 
    max_activity = (count/window_size) * 100
    # append the feature and label only if the count of max labels is greater than a certain low_limit for train data
    if is_train_data and max_activity>=low_limit: 
        features_list.append(features)
        labels_list.append(label)
    # append the feature and label only if the count of minimum labels is 0 for test and validation data 
    if (not is_train_data) and max_activity==100:
        features_list.append(features)
        labels_list.append(label)
       
        
  features_list = np.array(features_list)
  labels_list = np.expand_dims(np.array(labels_list), axis=1)
  return features_list, labels_list


# this functions has acc and gyr data combined for each body part with the activity labels as last column
def get_one_proband_data(filepath, bodypart):
  df=[]
  for l in labels:
    df_ind = read_individually(filepath,bodypart, l)
    if len(df_ind) != 0:
      df_ind = pd.concat(df_ind, axis=0, ignore_index=1)
      df.append(df_ind)
  return df

# For a single proband - this functions read acc data and gyr data for each bodypart and combined them in one dataframe. 
def read_individually(filepath, bodypart, label):
  df=[]
  file_list = os.listdir(filepath)
  file_list.sort()
  acc_files = [filename for filename in file_list if filename.startswith('acc') and filename.endswith('.csv')]
  gyr_files = [filename for filename in file_list if filename.startswith('Gyroscope') and filename.endswith('.csv')]
  for e,g in zip(acc_files, gyr_files):
    if bodypart in e and g:
      if label in e and g:
        acc_df = pd.read_csv(filepath+e)
        gyr_df = pd.read_csv(filepath+g)
        # acc_gyr_combined = pd.concat([acc_df, gyr_df], axis=1)
        acc_gyr_combined = pd.merge(acc_df, gyr_df, left_on='attr_time', right_on='attr_time')
        acc_gyr_combined['label'] = labels[label]
        df.append(acc_gyr_combined)
  return df



def write_as_tfrecord(features, labels, filepath):
   dataset = tf.data.Dataset.from_tensor_slices((features, labels))
   with tf.io.TFRecordWriter(filepath) as writer:
      for feature, label in dataset:
         feature = tf.io.serialize_tensor(feature)
         label =  tf.io.serialize_tensor(label)
         example_dict = {
         "feature":tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.numpy()])),
         "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.numpy()]))
        }
         example = tf.train.Example(features=tf.train.Features(feature=example_dict))
         writer.write(example.SerializeToString())


def resample_data(features, labels):
    features_resampled = np.empty((0, features.shape[1], features.shape[2]))
    labels_resampled = np.empty((0, labels.shape[1]))
    activities, activity_counts = np.unique(labels, return_counts=True)
    max_act = np.max(activity_counts)
    for activity in activities:
        activity_indices = np.where(labels == activity)[0]
        indices = np.random.choice(activity_indices, size=max_act, replace=True)
        labels_resampled = np.append(labels_resampled, labels[indices], axis=0)
        features_resampled = np.append(features_resampled, features[indices], axis=0)

    return features_resampled, labels_resampled 


@gin.configurable
def make_tfrecords_rwhar(data_dir, target_dir, window_length, shift, bodypart):
    if os.path.exists(target_dir):
        logging.info("[INFO] Records already exists")
        return 0
    file_list = os.listdir(data_dir)
    file_list.sort()
    train_users = [1,2,5,8,11,12,13,15]
    val_user = [3]
    test_user = [9,10]
    train_data = np.empty(shape=(0, window_length, 6))
    train_labels = np.empty(shape = (0,1))
    val_data = np.empty(shape=(0, window_length, 6))
    val_labels = np.empty(shape = (0,1))
    test_data = np.empty(shape=(0, window_length, 6))
    test_labels = np.empty(shape = (0,1))
    for proband in file_list:
        if 'proband' in proband:
            filepath = data_dir+proband+'/data/'
            user_id = int(''.join(c for c in proband if c.isdigit()))
            proband_data = get_one_proband_data(filepath, bodypart)
            data = pd.concat(proband_data, axis=0, ignore_index=True)
            data.columns = ["id_acc","attr_time","acc_x", "acc_y", "acc_z","id_gyr","gyr_x", "gyr_y", "gyr_z", "label"]
            data_cleaned = data.dropna()
            data_cleaned.drop(["id_acc", "attr_time", "id_gyr"],axis=1, inplace=True)
            to_normalize = data_cleaned.drop(["label"], axis=1)
            normalized_data = zscore(to_normalize, axis=0)
            normalized_data['label'] = data_cleaned['label']
            normalized_data = normalized_data.iloc[250:-250]
            is_train_data = (user_id in train_users)
            window_features, window_labels = window_maker(normalized_data, is_train_data, window_length, shift)
            if user_id in train_users:
              train_data= np.append(train_data, window_features, axis=0)
              train_labels= np.append(train_labels, window_labels, axis=0)
            if user_id in val_user:
              val_data= np.append(val_data, window_features, axis=0)
              val_labels= np.append(val_labels, window_labels, axis=0)
            if user_id in test_user:
              test_data= np.append(test_data, window_features, axis=0)
              test_labels= np.append(test_labels, window_labels, axis=0)
            print(f"Loaded user {user_id} with features shape {window_features.shape} labels shape {window_labels.shape}")  
    
    train_data, train_labels = resample_data(train_data, train_labels)

    train_data, train_labels = shuffle(train_data, train_labels)
    test_data, test_labels = shuffle(test_data, test_labels)
    val_data, val_labels = shuffle(val_data, val_labels)

    os.makedirs(target_dir)
    write_as_tfrecord(train_data, train_labels, target_dir+'/train.tfrecords')
    write_as_tfrecord(val_data, val_labels, target_dir+'/val.tfrecords')
    write_as_tfrecord(test_data, test_labels, target_dir+'/test.tfrecords')

    logging.info("[TF records Dataset Created] Details below...")
    logging.info(f"Train data: {train_data.shape}, Labels: {train_labels.shape}")
    logging.info(f"Val data: {val_data.shape}, Labels: {val_labels.shape}")
    logging.info(f"Test data: {test_data.shape}, Labels: {test_labels.shape}")

    return 1


