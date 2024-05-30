import tensorflow as tf
import gin

from models.architectures import cnn_se, cnn_1, transfer_model
from models.layers import ModeLayer

@gin.configurable
class EnsembleModel:
    
    def __init__(self, models, ckpt_paths, input_shape, type='hard'):
        self.type = type 
        self.model_list = self.load_models(models,ckpt_paths)
        self.input_shape = input_shape

    def load_models(self, models, ckpt_paths):
        model_list = []    
        for (model,ckpt_path) in zip(models,ckpt_paths):
            if model=='cnn_se':
                model_to_append = cnn_se()
            elif model=='cnn_1':
                model_to_append = cnn_1()
            elif model=='DenseNet121':
                model_to_append = transfer_model(base_model_name='DenseNet121', model_name='DenseNet121')
            elif model=='InceptionV3':
                model_to_append = transfer_model(base_model_name='InceptionV3', model_name='InceptionV3')
            elif model=='InceptionResnet':
                model_to_append = transfer_model(base_model_name='InceptionResnet', model_name='InceptionResnet')
            elif model=='VGG16':
                model_to_append = transfer_model(base_model_name='VGG16', model_name='VGG16')

            checkpoint = tf.train.Checkpoint(model=model_to_append)
            checkpoint.restore(tf.train.latest_checkpoint(ckpt_path))
            model_list.append(model_to_append)
        return model_list
    
    
    def __call__(self):
        if self.type=='hard':
            model = self.get_hard_ensemble_model()
        elif self.type=='soft':
            model = self.get_soft_ensemble_model()
        return model
    
    
    def get_hard_ensemble_model(self):
        """
        return: tensorflow model that returns the prediction after hard voting (mode)
        """
        input_layer = tf.keras.Input(shape=self.input_shape)

        logits = [model(input_layer, training=False) for model in self.model_list]
        softmax_output = [tf.nn.softmax(logit) for logit in logits]
        predictions = [tf.argmax(output, axis=-1) for output in softmax_output]
        predictions_stack = tf.stack(predictions, axis=-1)
        mode_layer = ModeLayer(num_classes=2)
        onehot_mode = mode_layer(predictions_stack)
        model = tf.keras.Model(inputs=input_layer, outputs=onehot_mode) 
        return model


    def get_soft_ensemble_model(self):
        """
        return: tensorflow model that returns the prediction after soft voting 
        """
        input_layer = tf.keras.Input(shape=self.input_shape)
        logits = [model(input_layer, training=False) for model in self.model_list]
        softmax_output = [tf.nn.softmax(logit) for logit in logits]
        softmax_stacked = tf.stack(softmax_output, axis=-1)
        softmax_prediction = tf.reduce_mean(softmax_stacked, axis=-1)

        model = tf.keras.Model(inputs=input_layer, outputs=softmax_prediction) 
        return model
    



            
