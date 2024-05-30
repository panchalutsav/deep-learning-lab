import gin
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import os
import cv2


@gin.configurable
class GradCam:
    def __init__(self, model, layer_name,class_idx=None):
        self.model = model
        self.layer_name = layer_name
        self.class_idx = class_idx
        self.grad_model = tf.keras.models.Model(self.model.inputs,
                                                [self.model.get_layer(name=self.layer_name).output, self.model.output])
    
    @tf.function
    def get_average_grads(self, image):
        with tf.GradientTape() as tape:
            layer_activations, preds = self.grad_model(image)
            if self.class_idx is None:
                self.class_idx = tf.argmax(preds[0])
            class_channel = preds[:, self.class_idx]
            
        grads = tape.gradient(class_channel, layer_activations)
        average_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        return average_grads, layer_activations


    def get_heatmap(self, image):
        average_grads, layer_activations = self.get_average_grads(image)
        average_grads = tf.expand_dims(average_grads, axis=-1)
        heatmap = layer_activations @ average_grads
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()


    def get_jet_heatmap(self, heatmap, image):
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
        jet_heatmap = jet_heatmap/jet_heatmap.max()
        return jet_heatmap


    def apply_gradcam_one_image(self, image , alpha = 0.5):
        ds_image = tf.expand_dims(image, axis=0)
        heatmap = self.get_heatmap(ds_image)
        ds_image = tf.squeeze(ds_image, axis=0)
        jet_heatmap = self.get_jet_heatmap(heatmap, ds_image)
        superimposed_img = jet_heatmap * alpha + ds_image * (1-alpha)
        return superimposed_img


    # ----------deprecated: apply gradcam for whole list of images: use above function instead that applys gradcam for one image
    def apply_gradcam(self, image_list, save_dir, alpha=0.5):
        for i, (ds_image, label) in enumerate(self.dataset):
            img_idx = image_list[i]
            original_image = np.array(self.original_images[i])
            original_image = original_image/255.0
            image_name = f"img_{img_idx}_label-{label}.jpg"
            save_path = os.path.join(save_dir, image_name)

            ds_image = tf.expand_dims(ds_image, axis=0)
            heatmap = self.get_heatmap(ds_image)
            ds_image = tf.squeeze(ds_image, axis=0)
            jet_heatmap = self.get_jet_heatmap(heatmap, ds_image)
            superimposed_img = jet_heatmap * alpha + ds_image * (1-alpha)

            total_image = tf.concat([original_image, superimposed_img], axis=0)
            yield total_image, image_name, save_path
