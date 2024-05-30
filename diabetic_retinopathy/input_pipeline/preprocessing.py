import gin
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""
    # Image Normalization.
    max = tf.reduce_max(image)
    image = tf.cast(image, tf.float32) / tf.cast(max, tf.float32)
    return image, label


for i in range(3):
    seed = (i, 0)  # tuple of size (2,)


def augment(image, label):
    """Data augmentation"""
    image = apply_randomly(image, augment_contrast)
    image = apply_randomly(image, augment_saturation)
    image = apply_randomly(image, augment_brightness)
    image = apply_randomly(image, augment_flip_left_right)
    image = apply_randomly(image, augment_flip_updown)
    image = apply_randomly(image, augment_random_crop)
    image = apply_randomly(image, random_rotate)
    return image, label


@gin.configurable
def apply_randomly(img, apply_func, p=0.3):
    if tf.random.uniform([]) < p:
        img = apply_func(img)
    else:
        img = img
    return img


def augment_brightness(image):
    img = tf.image.stateless_random_brightness(image, 0.3, seed=seed)
    return img


def augment_contrast(image):
    img = tf.image.stateless_random_contrast(image, 0.2, 1, seed=seed)
    return img


def augment_saturation(image):
    img = tf.image.stateless_random_saturation(image, 0.1, 1.0, seed=seed)
    return img


def augment_flip_left_right(image):
    img = tf.image.stateless_random_flip_left_right(image, seed=seed)
    return img


def augment_flip_updown(image):
    img = tf.image.stateless_random_flip_up_down(image, seed=seed)
    return img


def augment_random_crop(image):
    h = gin.query_parameter('preprocess.img_height')
    w = gin.query_parameter('preprocess.img_width')
    cropped_h = int(h / 1.3)
    cropped_w = int(w / 1.3)
    img = tf.image.stateless_random_crop(image, (cropped_h, cropped_w, 3), seed=seed)
    img = tf.image.resize(img, [w, h])
    return img

def random_rotate(image):
    random_angle = tf.random.uniform([],minval=-0.5,maxval=0.5, seed=14) * np.pi
    img = tfa.image.rotate(image,
                             angles=random_angle,
                             )
    return img
