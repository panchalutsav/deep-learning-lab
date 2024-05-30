"""
https://medium.com/@chinesh4/generalized-way-of-interpreting-cnns-a7d1b0178709
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        last_conv_layer_name = list(filter(lambda x: isinstance(x, keras.layers.Conv2D), self.model.layers))[-1].name
        self.backprop_model = tf.keras.models.Model(self.model.inputs,[self.model.get_layer(name=last_conv_layer_name).output])
        self.backprop_model = self.change_activations(model=self.backprop_model) 
                
    def prep_input(self, img):
        image = tf.expand_dims(img, axis=0)
        image = tf.cast(image, tf.float32)
        return image

    @tf.custom_gradient
    def guidedRelu(self, x):
        def grad(dy):
            return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
        return tf.nn.relu(x), grad

    def change_activations(self, model):
        layer_dict = [layer for layer in model.layers[1:] if hasattr(layer,'activation')]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = self.guidedRelu
        return model

    def deprocess_image(self,grad):
        grad = grad.copy()

        # normalize the tensor
        grad -= grad.mean()
        grad /= (grad.std() + K.epsilon())
        grad*=0.25

        # clip to [0, 1]
        grad+=0.5
        grad = np.clip(grad, 0, 1)

        # convert to RGB array
        grad*=255
        grad = grad.astype('uint8')
        #grad = np.clip(grad, 0, 255).astype('uint8')
        return grad

    def apply_guidedbackprop_one_image(self, image):
        input_img = self.prep_input(image)
        with tf.GradientTape() as tape:
            tape.watch(input_img)
            result = self.backprop_model(input_img)
        grads = tape.gradient(result, input_img)[0]
        deprocessed_img = self.deprocess_image(np.array(grads))
        return deprocessed_img