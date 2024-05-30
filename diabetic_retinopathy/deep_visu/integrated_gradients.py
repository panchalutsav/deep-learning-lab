import gin
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import os
import cv2
import logging
import matplotlib.pyplot as plt


class Integrated_gradients:
    def __init__(self, model):
        self.model = model 
        self.baseline = tf.zeros([256,256,3])
        self.m_steps = 240
        self.alphas = tf.linspace(start=0.0, stop=1.0, num=self.m_steps+1)

    def interpolate_images(self, baseline,
                       image,
                       alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - baseline_x
        images = baseline_x +  alphas_x * delta
        return images
    
    def compute_gradients(self, images):
        with tf.GradientTape() as tape:
          tape.watch(images)
          preds = self.model(images)
        return tape.gradient(preds, images)
    
    def integral_approximation(self, gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients
    
    def plot_img_attributions(self, baseline,
                          image,
                          m_steps=50,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.4):
        # Sum of the attributions across color channels for visualization.
        # The attribution mask shape is a grayscale image with height and width
        # equal to the original image.
        attributions = self.integrated_gradients(baseline=baseline,
                                      image=image,
                                      m_steps=m_steps)
        attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
        fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

        axs[0, 0].set_title('Baseline image')
        axs[0, 0].imshow(baseline)
        axs[0, 0].axis('off')

        axs[0, 1].set_title('Original image')
        axs[0, 1].imshow(image)
        axs[0, 1].axis('off')

        axs[1, 0].set_title('Attribution mask')
        axs[1, 0].imshow(attribution_mask, cmap=cmap)
        axs[1, 0].axis('off')

        axs[1, 1].set_title('Overlay')
        axs[1, 1].imshow(attribution_mask, cmap=cmap)
        axs[1, 1].imshow(image, alpha=overlay_alpha)
        axs[1, 1].axis('off')

        #plt.tight_layout()
        #plt.savefig("int_grads.png")
        return attribution_mask
    
    def integrated_gradients(self, baseline,
                         image,
                         m_steps=50,
                         batch_size=32):
         # Generate alphas.
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

        # Collect gradients.    
        gradient_batches = []

        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for alpha in tf.range(0, len(alphas), batch_size):
          from_ = alpha
          to = tf.minimum(from_ + batch_size, len(alphas))
          alpha_batch = alphas[from_:to]

          gradient_batch = self.one_batch(baseline, image, alpha_batch)
          gradient_batches.append(gradient_batch)

        # Concatenate path gradients together row-wise into single tensor.
        total_gradients = tf.concat(gradient_batches, axis=0)

        # Integral approximation through averaging gradients.
        avg_gradients = self.integral_approximation(gradients=total_gradients)

        # Scale integrated gradients with respect to input.
        integrated_gradients = (image - baseline) * avg_gradients

        return integrated_gradients

    @tf.function
    def one_batch(self, baseline, image, alpha_batch):
        # Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = self.interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

        # Compute gradients between model outputs and interpolated inputs.
        gradient_batch = self.compute_gradients(images=interpolated_path_input_batch)
        return gradient_batch
    
    def apply_intgrad_one_image(self, image):
        image = tf.image.convert_image_dtype(image, tf.float32, saturate=False, name=None)
        attribution_mask = self.plot_img_attributions(image=image,
                          baseline=self.baseline,
                          m_steps=self.m_steps,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.4)
        return attribution_mask







            



            
        
            
    




