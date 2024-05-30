import os
import sys

import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import logging
import gin
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sys

seed = 100


@gin.configurable
def make_tfrecords(data_dir, target_dir):
    LABELS_PATH = data_dir + "labels/"
    IMAGES_PATH = data_dir + "images/"

    np.random.seed(seed)

    if os.path.exists(target_dir):
        return 0

    df_train_val = pd.read_csv(LABELS_PATH + "train.csv", usecols=['Image name', 'Retinopathy grade'])
    df_test = pd.read_csv(LABELS_PATH + "test.csv", usecols=['Image name', 'Retinopathy grade'])

    convert_to_binary(df_train_val)
    convert_to_binary(df_test)

    df_train, df_val = train_test_split(df_train_val, test_size=0.2, shuffle=True)

    df_train = resample_(df_train)
   # df_val = resample_(df_val)

    if not (os.path.isdir(IMAGES_PATH + 'train/') or IMAGES_PATH + 'test/'):
        logging.error(f"Path does not exist: {IMAGES_PATH}train/")
        logging.error(f"Path does not exist: {IMAGES_PATH}test/")
        sys.exit()

    os.makedirs(target_dir)
    record_writer(df_train, target_dir + "train.tfrecords", IMAGES_PATH + 'train/')
    record_writer(df_val, target_dir + "validation.tfrecords", IMAGES_PATH + 'train/')
    record_writer(df_test, target_dir + "test.tfrecords", IMAGES_PATH + 'test/')
    logging.info(f"TF records created at {target_dir}")

    return 1


def record_writer(df, record_path, images_path):
    with tf.io.TFRecordWriter(record_path) as writer:
        for index, row in df.iterrows():
            image = cv2.imread(images_path + row["Image name"] + ".jpg")
            preprocessed_image = preprocess_image(image)
            label = row["Retinopathy grade"]
            tf_example = image_example(preprocessed_image, label)
            writer.write(tf_example.SerializeToString())

        writer.close()


def image_example(image, label):
    """ Create the features dictionary - Adapted from tensorflow docs"""
    string_image = cv2.imencode('.jpg', image)[1].tobytes()
    feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image_raw': bytes_feature(string_image)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#crop and resize the image
def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    pos = np.nonzero(threshold)
    right_boundary = pos[1].max()
    left_boundary = pos[1].min()
    image = image[:, left_boundary:right_boundary]
    # computations to obtain square image (padding at desired positions)
    upper_diff = (image.shape[1] - image.shape[0]) // 2
    lower_diff = image.shape[1] - image.shape[0] - upper_diff
    image = cv2.copyMakeBorder(image, upper_diff, lower_diff, 0, 0, cv2.BORDER_CONSTANT)

    return cv2.resize(image, (256, 256))

def apply_clahe(image):
    clahe_model = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_b = clahe_model.apply(image[:,:,0])
    image_g = clahe_model.apply(image[:,:,1])
    image_r = clahe_model.apply(image[:,:,2])
    image_clahe = np.stack((image_b,image_g,image_r), axis=2)
    return image_clahe

@gin.configurable
def preprocess_image(image, with_clahe=False, with_bens=False):
    img_size = 256
    image = crop_image(image)
    if with_clahe:
        image = apply_clahe(image)
    if with_bens:
        image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) ,img_size/30) ,-4 ,128)

    return image


def convert_to_binary(df):
    df.loc[df['Retinopathy grade'] < 2, "Retinopathy grade"] = 0
    df.loc[df['Retinopathy grade'] > 1, "Retinopathy grade"] = 1
    return df


# we will resample only the minority class
def resample_(df):
    max_idx = df['Retinopathy grade'].value_counts().idxmax()
    total_samples = df['Retinopathy grade'].value_counts().max()
    df0 = df.loc[df['Retinopathy grade'] == 0]
    df1 = df.loc[df['Retinopathy grade'] == 1]
    df0_resampled = df0
    df1_resampled = df1

    if max_idx == 1:
        df0_resampled = resample(df0, replace=True, n_samples=total_samples, random_state=seed)
    elif max_idx == 0:
        df1_resampled = resample(df1, replace=True, n_samples=total_samples, random_state=seed)

    df = pd.concat([df0_resampled, df1_resampled])

    return df
