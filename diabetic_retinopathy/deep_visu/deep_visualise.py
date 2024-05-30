import os
import gin
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import logging
import wandb

from datetime import datetime
from deep_visu.grad_cam import GradCam
from deep_visu.integrated_gradients import Integrated_gradients
from deep_visu.guided_backprop import GuidedBackprop
from input_pipeline.tfrecords import preprocess_image, convert_to_binary
from input_pipeline.preprocessing import preprocess
import matplotlib.pyplot as plt

@gin.configurable
class DeepVisualize:
    def __init__(self, model, run_paths, data_dir, target_dir, layer_name, image_list_test=None, image_list_train=None, chkpt=False, log_wandb=False):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.run_paths = run_paths
        self.model = model
        self.layer_name = layer_name
        self.image_list_test = image_list_test
        self.image_list_train = image_list_train
        self.chkpt = chkpt

        if (image_list_train is None) or (image_list_test is None):
            logging.info("No images specified from either test or train set. Specify files from both. Exiting run.")
            sys.exit(0)


    # creates specific dataset which contains only labelled images from config.gin
    def create_dataset(self):
        labels_path = self.data_dir + "labels/"
        images_path = self.data_dir + "images/"

        df_train = pd.read_csv(labels_path + "train.csv", usecols=['Image name', 'Retinopathy grade'])
        df_test = pd.read_csv(labels_path + "test.csv", usecols=['Image name', 'Retinopathy grade'])
        df_train = df_train.sort_values(by='Image name')
        df_test = df_test.sort_values(by='Image name')

        train_idx = [i - 1 for i in self.image_list_train]
        test_idx = [i - 1 for i in self.image_list_test]

        df_train = df_train.iloc[train_idx]
        df_test = df_test.iloc[test_idx]

        file_paths_train = [(images_path + "train/" + filename + ".jpg") for filename in df_train['Image name']]
        file_paths_test = [(images_path + "test/" + filename + ".jpg") for filename in df_test['Image name']]
        ds_images_train = [self.load(file_path, with_preprocess=True) for file_path in file_paths_train]
        ds_images_test = [self.load(file_path, with_preprocess=True) for file_path in file_paths_test]

        convert_to_binary(df_train)
        convert_to_binary(df_test)
        labels_train = df_train['Retinopathy grade'].values
        labels_test = df_test['Retinopathy grade'].values

        ds_train = tf.data.Dataset.from_tensor_slices((ds_images_train, labels_train))
        ds_test = tf.data.Dataset.from_tensor_slices((ds_images_test, labels_test))

        #load actual images for comparisons 
        images_train = [self.load(file_path, with_preprocess=False) for file_path in file_paths_train]
        images_test = [self.load(file_path, with_preprocess=False) for file_path in file_paths_test]

        return ds_train, ds_test, images_train, images_test

    def load(self, img_path, with_preprocess):
        image = cv2.imread(img_path)
        if with_preprocess:
            image = preprocess_image(image)
        else:
            image = preprocess_image(image, with_clahe=False, with_bens=False)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    

    def plot(self, img,grad_img, int_grad_img,guided_bp ,label, path):
        fig, axs = plt.subplots(nrows=2, ncols=3, squeeze=False, figsize=(8, 8))
        fig.suptitle(f"Deep Viz outputs\n Label: {label}", fontsize=10)

        axs[0, 0].set_title('Original Image')
        axs[0, 0].imshow(img)
        axs[0, 0].axis('off')

        axs[0, 1].set_title('Gradcam')
        axs[0, 1].imshow(grad_img)
        axs[0, 1].axis('off')

        axs[0, 2].set_title('Guided Backprop')
        axs[0, 2].imshow(guided_bp)
        axs[0, 2].axis('off')

        axs[1, 0].set_title('Integrated Gradient')
        axs[1, 0].imshow(int_grad_img, cmap=plt.cm.inferno)
        axs[1, 0].axis('off')

        axs[1, 1].set_title('Integrated Gradient (overlay)')
        axs[1, 1].imshow(int_grad_img, cmap=plt.cm.inferno)
        axs[1, 1].imshow(img, alpha=0.4)
        axs[1, 1].axis('off')

        axs[1, 2].axis('off')
  

        plt.tight_layout()
        plt.savefig(path)
        return fig


    def visualize(self):
        logging.info("\n===============Starting Deep Visualisation================")
        ds_train, ds_test, images_train, images_test = self.create_dataset()
        ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        logging.info("dataset created from image list")

        checkpoint = tf.train.Checkpoint(model=self.model)
        if self.chkpt:
            checkpoint.restore(tf.train.latest_checkpoint(self.chkpt))
            logging.info(f"model loaded with checkpoint from {self.chkpt}")
        else:
            checkpoint.restore(tf.train.latest_checkpoint(self.run_paths['path_ckpts_train']))
            logging.info(f"model loaded with checkpoint from {self.run_paths['path_ckpts_train']}")

        output_dir = os.path.join(self.target_dir, 'deepviz_output')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        out_dir = f"out_at_{timestamp}"
        deepviz_out_dir = self.target_dir + "deepviz_output/" + f"{out_dir}/"
        os.makedirs(deepviz_out_dir)

        train_dir = os.path.join(deepviz_out_dir, 'train')
        os.makedirs(train_dir)
        test_dir = os.path.join(deepviz_out_dir, 'test')
        os.makedirs(test_dir)

        gradcam = GradCam(self.model, self.layer_name)
        int_grad = Integrated_gradients(self.model)
        guided_backprop = GuidedBackprop(self.model)
        
        # deep viz for train_images
        for i, (image, label) in enumerate(ds_train):
            img_name = f"deep_viz_{self.image_list_train[i]}"
            grad_img = gradcam.apply_gradcam_one_image(image)
            int_grad_img = int_grad.apply_intgrad_one_image(image)
            guided_backprop_img = guided_backprop.apply_guidedbackprop_one_image(image)
            save_path = os.path.join(train_dir, img_name)
            _ = self.plot(image, grad_img=grad_img, int_grad_img=int_grad_img, guided_bp = guided_backprop_img,\
                          label=label, path= save_path)

        # deep viz for test images
        for i, (image, label) in enumerate(ds_test):
            img_name = f"deep_viz_{self.image_list_test[i]}"
            grad_img = gradcam.apply_gradcam_one_image(image)
            intgrad_img = int_grad.apply_intgrad_one_image(image)
            guided_backprop_img = guided_backprop.apply_guidedbackprop_one_image(image)
            save_path = os.path.join(test_dir, img_name)
            _ = self.plot(image, grad_img=grad_img, int_grad_img=intgrad_img,guided_bp = guided_backprop_img,\
                          label=label, path=save_path)

        logging.info(f"images saved in {deepviz_out_dir}")
        logging.info("\n=============== Deep Visualisation Completed ================")
        
