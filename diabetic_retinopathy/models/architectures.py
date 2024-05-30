import gin
import keras
import tensorflow as tf
import logging

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Add, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import layers, Sequential


from models.layers import vgg_block, cnn_block, skip_connect, squeeze_excite_block


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


@gin.configurable
def cnn_1(input_shape, filters, kernel_size, strides, pool_size, dropout_rate, batch_norm_blocks=False, maxpool_blocks=False,  dropout_blocks=False, skip_connection_pairs=False, SE_blocks=False):
    
    inputs = Input(shape=input_shape)

    # convolutional layers
    out = []
    for block in range(len(filters)):

        input_layer = inputs if block==0 else out[-1]
        #Conv2d
        out_block = cnn_block(input_layer, 
                        filters[block], 
                        kernel_size, 
                        strides[block]
                        )
        out.append(out_block)

        #Skip connection
        if skip_connection_pairs:
            skip_connect(out, block, skip_connection_pairs)

        #batch normalisation, dropout_blocks, maxpooling
        if batch_norm_blocks and (block in batch_norm_blocks):
            out[-1] = BatchNormalization()(out[-1])
        if maxpool_blocks and (block in maxpool_blocks):
            out[-1] = MaxPool2D(pool_size)(out[-1])
        if dropout_blocks and (block in dropout_blocks): 
            out[-1] = Dropout(dropout_rate)(out[-1])

    # dense layers
    out_dense = GlobalAveragePooling2D()(out[-1])
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=32, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0), activation='relu',
                      kernel_initializer =tf.keras.initializers.HeNormal()
                      )(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=16, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0), activation='relu',
                      kernel_initializer =tf.keras.initializers.HeNormal()
                      )(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=4, 
                      kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0)
                      )(out_dense)

    outputs = Dense(units=2)(out_dense)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_1")
    model.build(input_shape=input_shape)

    logging.info(f"cnn_1 input shape:  {model.input_shape}")
    logging.info(f"cnn_1 output shape: {model.output_shape}")

    return model



@gin.configurable
def cnn_se(input_shape, filters, kernel_size, strides, pool_size, dropout_rate, batch_norm_blocks=False,
           maxpool_blocks=False,  dropout_blocks=False, skip_connection_pairs=False, SE_blocks=False):
    
    inputs = Input(shape=input_shape)

    # convolutional layers
    out = []
    for block in range(len(filters)):

        input_layer = inputs if block==0 else out[-1]
        #Conv2d
        out_block = cnn_block(input_layer, 
                        filters[block], 
                        kernel_size, 
                        strides[block]
                        )
        out.append(out_block)

        if SE_blocks:
            out[-1] = squeeze_excite_block(out[-1])

        #Skip connection
        if skip_connection_pairs:
            skip_connect(out, block, skip_connection_pairs)

        #batch normalisation, dropout_blocks, maxpooling
        if batch_norm_blocks and (block in batch_norm_blocks):
            out[-1] = BatchNormalization()(out[-1])
        if maxpool_blocks and (block in maxpool_blocks):
            out[-1] = MaxPool2D(pool_size)(out[-1])
        if dropout_blocks and (block in dropout_blocks): 
            out[-1] = Dropout(dropout_rate)(out[-1])

    # dense layers
    out_dense = GlobalAveragePooling2D()(out[-1])
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=32, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0), activation='relu',
                      #kernel_initializer =tf.keras.initializers.HeNormal()
                      )(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=16, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0), activation='relu',
                      #kernel_initializer =tf.keras.initializers.HeNormal()
                      )(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=4, 
                      #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0)
                      )(out_dense)

    outputs = Dense(units=2)(out_dense)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_se")
    model.build(input_shape=input_shape)

    logging.info(f"cnn_se input shape:  {model.input_shape}")
    logging.info(f"cnn_se output shape: {model.output_shape}")

    return model


@gin.configurable
def transfer_model(input_shape, dense_units, dropout_rate,base_model_name="DenseNet121", model_name = 'transfer_model'):

    inputs = Input(shape=input_shape)
    if base_model_name == 'InceptionResnet':
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                                weights="imagenet", 
                                                                input_shape=input_shape, 
                                                                pooling=None)
    elif base_model_name == 'InceptionV3':
        base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                                weights="imagenet", 
                                                                input_shape=input_shape, 
                                                                pooling=None)
    elif base_model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(include_top=False,
                                                                weights="imagenet", 
                                                                input_shape=input_shape, 
                                                                pooling=None)

    elif base_model_name == 'DenseNet121':
        base_model = tf.keras.applications.DenseNet121(include_top=False,
                                                                weights="imagenet", 
                                                                input_shape=input_shape, 
                                                                pooling=None)
    
    out = base_model(inputs)

    base_model.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    out_dense = GlobalAveragePooling2D()(out)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=int(dense_units/2), kernel_regularizer=regularizers.l2(0.01), activation='relu')(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)
    out_dense = Dense(units=int(dense_units/4), kernel_regularizer=regularizers.l2(0.01))(out_dense)
    out_dense = Dropout(dropout_rate)(out_dense)

    outputs = Dense(units=2)(out_dense)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    logging.info(f"transfer_model input shape:  {model.input_shape}")
    logging.info(f"transfer_model output shape: {model.output_shape}")

    return model

