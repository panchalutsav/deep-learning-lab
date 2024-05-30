import gin
import keras
import tensorflow as tf
import logging

from tensorflow.keras import regularizers
from tensorflow.keras import layers
from models.layers import one_lstm_layer, one_bidirectional_lstm_layer, one_gru_layer
from tensorflow import keras
from tensorflow.keras import layers



@gin.configurable
def model_bidirectional_LSTM(window_length, num_lstm, dense_units, lstm_cells, n_classes, dropout_rate=0.3):
    model = keras.Sequential([keras.Input(shape=(window_length, 6))])
    
    for e in range(num_lstm):
        layer = one_bidirectional_lstm_layer(lstm_cells, dropout_rate=dropout_rate)
        model.add(layer)
        model.add(layers.BatchNormalization())
    
    model.add(layers.Bidirectional(layers.LSTM(lstm_cells, dropout=dropout_rate)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(n_classes, activation="linear"))
    
    model.build()
    print(model.input_shape)
    print(model.output_shape)

    return model


@gin.configurable
def model1_LSTM(window_length,num_lstm,dense_units,lstm_cells,n_classes,dropout_rate=0.3):
    model = keras.Sequential([keras.Input(shape=(window_length,6), dtype=tf.float32)])
    for e in range(num_lstm):
        layer = one_lstm_layer(lstm_cells, dropout_rate=dropout_rate)
        model.add(layer)
        model.add(layers.BatchNormalization())
    model.add(layers.LSTM(lstm_cells, dropout=dropout_rate))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(dense_units , activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(n_classes, activation="linear"))
    model.build()
    print(model.input_shape)
    print(model.output_shape)


    return model


@gin.configurable
def model1_GRU(window_length, num_gru, dense_units, gru_cells, n_classes, dropout_rate=0.3):
    model = keras.Sequential([keras.Input(shape=(window_length, 6))])
    
    for e in range(num_gru):
        layer = one_gru_layer(gru_cells, dropout_rate=dropout_rate)
        model.add(layer)
        model.add(layers.BatchNormalization())
    
    model.add(layers.GRU(gru_cells, dropout=dropout_rate))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(n_classes, activation="linear"))
    
    model.build()
    print(model.input_shape)
    print(model.output_shape)
    
    return model

@gin.configurable
def model1D_Conv(window_length, num_conv_layers, filters, kernel_size, dense_units, n_classes, dropout_rate=0.3):
    model = keras.Sequential([keras.Input(shape=(window_length, 6), dtype=tf.float64)])

    for _ in range(num_conv_layers):
        model.add(layers.Conv1D(filters, kernel_size, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.GlobalMaxPooling1D())  # Use GlobalMaxPooling1D to reduce the spatial dimensions

    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(n_classes, activation="linear"))

    model.build()
    print(model.input_shape)
    print(model.output_shape)

    return model






