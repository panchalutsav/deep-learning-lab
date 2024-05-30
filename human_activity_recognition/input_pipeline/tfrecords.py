import gin, logging, os, sys
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from scipy.stats import zscore, mode
import numpy as np
from sklearn.utils import shuffle
import re
import matplotlib.pyplot as plt

from input_pipeline.visualise_har import visualise_HAR



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

def resample_data(features, labels):
    features_resampled = np.empty((0, features.shape[1], features.shape[2]))
    labels_resampled = np.empty((0, labels.shape[1]))
    activities, activity_counts = np.unique(labels, return_counts=True)
    max_activity = np.max(activity_counts)
    for activity in activities:
        activity_indices = np.where(labels == activity)[0]
        indices = np.random.choice(activity_indices, size=max_activity, replace=True)
        labels_resampled = np.append(labels_resampled, labels[indices], axis=0)
        features_resampled = np.append(features_resampled, features[indices], axis=0)

    return features_resampled, labels_resampled 

   
@gin.configurable
def make_tfrecords(data_dir, target_dir, window_length, shift, visualise_expt):
    if visualise_expt:
       visualise_HAR(visualise_expt)

    if os.path.exists(target_dir):
        logging.info("[INFO] Records already exists")
        return 0
    if not os.path.exists(os.path.join(data_dir, "labels.txt")):
        logging.error("[ERROR] No Labels, check path again")
        return 0
    
    file_list = os.listdir(data_dir)
    file_list.sort()
    acc_data_list = []
    gyro_data_list = []
    for filename in file_list:
        filepath = data_dir+filename
        if 'acc' in filename:
            acc_data_list.append(filepath)
        if 'gyro' in filename:
            gyro_data_list.append(filepath)
        if 'label' in filename:
            label_data_path = filepath

    # read labels
    columns = ['expid', 'userid', 'actid', 'start', 'end']
    labels_df = pd.read_csv(label_data_path, delimiter=' ', header=None, names=columns)

    train_data = np.empty(shape=(0, window_length, 6))
    train_labels = np.empty(shape = (0,1))
    val_data = np.empty(shape=(0, window_length, 6))
    val_labels = np.empty(shape = (0,1))
    test_data = np.empty(shape=(0, window_length, 6))
    test_labels = np.empty(shape = (0,1))

    # loop over each file, normalize it, create windows and append it to arrays
    for e in range(len(acc_data_list)):
       
       acc_data = pd.read_csv(acc_data_list[e], delimiter=" ", header=None)
       gyro_data = pd.read_csv(gyro_data_list[e], delimiter=" ", header=None)
       combined_data = pd.concat([acc_data, gyro_data], axis=1)
       combined_data.columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
       pattern = re.compile(r'/home/data/HAPT_dataset/RawData/acc_exp(\d+)_user(\d+).txt')

        # Use the pattern to match against the filename
       match = pattern.match(acc_data_list[e])

       if match:
        exp_number = match.group(1)
        user_number = match.group(2)
       else:
        logging.error("[ERROR] Filename format does not match the expected pattern.")

       is_train_data = int(user_number) in range(1,22)
       is_val_data = int(user_number) in range(28,31)
       is_test_data = int(user_number) in range(22,28)
       normalized_data = zscore(combined_data, axis=0)
       normalized_data["label"] = 0

       labels = labels_df[(labels_df['expid'] == int(exp_number)) & (labels_df['userid'] == int(user_number))]
       for index, (actid, start, end) in labels[['actid', 'start', 'end']].iterrows():
            normalized_data.loc[start:end, "label"] = actid

       # remove first 5 seconds of data
       normalized_data = normalized_data.iloc[250:-250]
       
       # create windows and shift 
       window_features, window_labels = window_maker(normalized_data, is_train_data, window_length, shift) # overlapping as per config.gin
    
       if is_train_data:
        train_data = np.append(train_data, window_features, axis=0)
        train_labels = np.append(train_labels, window_labels, axis=0)
       elif is_test_data:
        test_data = np.append(test_data, window_features, axis=0)
        test_labels = np.append(test_labels, window_labels, axis=0)
       elif is_val_data:
        val_data = np.append(val_data, window_features, axis=0)
        val_labels = np.append(val_labels, window_labels, axis=0)

    # delete unlabelled data 
    train_data, train_labels = delete_no_activity(train_data, train_labels)
    test_data, test_labels = delete_no_activity(test_data, test_labels)
    val_data, val_labels = delete_no_activity(val_data, val_labels)

    # resample data
    train_data, train_labels = resample_data(train_data, train_labels)

    # shuffle
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
   
def delete_no_activity(features, labels):
  idxs = np.where(labels == 0)[0]
  features = np.delete(features, idxs, axis=0)
  labels = np.delete(labels, idxs, axis=0)
  return features, labels

    
    

