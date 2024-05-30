import tensorflow as tf
import gin

from models.architectures import model_bidirectional_LSTM, model1_LSTM, model1_GRU, model1D_Conv
from models.layers import ModeLayer

@gin.configurable
class EnsembleModel:
    
    def __init__(self, models, ckpt_paths, window_length, n_classes, type='hard'):
        self.type = type 
        self.window_length = window_length
        self.n_classes = n_classes
        self.model_list = self.load_models(models,ckpt_paths)

    def load_models(self, models, ckpt_paths):
        model_list = []    
        for (model,ckpt_path) in zip(models,ckpt_paths):
            if model=='model_bidirectional_LSTM':
                model_to_append = model_bidirectional_LSTM(window_length=self.window_length, n_classes=self.n_classes)
            elif model=='model1_LSTM':
                model_to_append = model1_LSTM(window_length=self.window_length, n_classes=self.n_classes)
            elif model=='model1_GRU':
                model_to_append = model1_GRU(window_length=self.window_length, n_classes=self.n_classes)
            elif model=='model1D_Conv':
                model_to_append = model1D_Conv(window_length=self.window_length, n_classes=self.n_classes)

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
        input_layer = tf.keras.Input(shape=(self.window_length, 6))

        logits = [model(input_layer, training=False) for model in self.model_list]
        softmax_output = [tf.nn.softmax(logit) for logit in logits]
        predictions = [tf.argmax(output, axis=-1) for output in softmax_output]
        predictions_stack = tf.stack(predictions, axis=-1)
        mode_layer = ModeLayer(num_classes=self.n_classes)
        onehot_mode = mode_layer(predictions_stack)
        model = tf.keras.Model(inputs=input_layer, outputs=onehot_mode) 
        return model


    def get_soft_ensemble_model(self):
        """
        return: tensorflow model that returns the prediction after soft voting 
        """
        input_layer = tf.keras.Input(shape=(self.window_length, 6))
        logits = [model(input_layer, training=False) for model in self.model_list]
        softmax_output = [tf.nn.softmax(logit) for logit in logits]
        softmax_stacked = tf.stack(softmax_output, axis=-1)
        softmax_prediction = tf.reduce_mean(softmax_stacked, axis=-1)

        model = tf.keras.Model(inputs=input_layer, outputs=softmax_prediction) 
        return model
    



            
