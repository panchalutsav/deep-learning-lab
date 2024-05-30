import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, GRU


def one_lstm_layer(lstm_cells, dropout_rate):
    return LSTM(lstm_cells, dropout=dropout_rate,return_sequences=True)

def one_bidirectional_lstm_layer(lstm_cells, dropout_rate):
    return Bidirectional(LSTM(lstm_cells, dropout=dropout_rate, return_sequences=True))

def one_gru_layer(gru_cells, dropout_rate):
    return GRU(gru_cells, dropout=dropout_rate, return_sequences=True)

class ModeLayer(Layer):
    def __init__(self, num_classes):
        super(ModeLayer, self).__init__()
        self.num_classes = num_classes


    def call(self, inputs):
        def find_mode(element_of_batch):
            unique, _, count = tf.unique_with_counts(element_of_batch)
            mode_idx = tf.argmax(count)
            mode = unique[mode_idx]
            mode_one_hot = tf.one_hot(mode, depth=self.num_classes)
            return mode_one_hot
        
        modes = tf.map_fn(find_mode, inputs, dtype=tf.float32)
        return modes