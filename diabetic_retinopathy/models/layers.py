import gin
import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Add, Multiply, Layer

@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out


def cnn_block(input, filter, kernel_size, stride):
    conv2D = Conv2D(filters=filter,
                    kernel_size=kernel_size,
                    padding='same',
                    strides=stride,
                    kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0),
                    kernel_initializer =tf.keras.initializers.HeNormal()
                    )
    out = conv2D(input)
    out = Activation('relu')(out)
    return out


def skip_connect(out, block, skip_connection_pairs):
    for skip_connection in skip_connection_pairs:
        if block == skip_connection[1]:
            out_to_add = out[skip_connection[0]]
            out_to_add = Activation('relu')(out_to_add)
            #reshape for equal number of feature maps if required
            if not (out[-1].shape[-1] == out_to_add.shape[-1]):
                out_to_add = Conv2D(filters=out[-1].shape[-1], 
                                    kernel_size=(1,1), 
                                    #kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)
                                    )(out_to_add)
                out_to_add = Activation('relu')(out_to_add)
            out[-1] = Add()([out_to_add, out[-1]])


def squeeze_excite_block(input, ratio=8):
    num_channels = int(input.shape[-1])
    se = GlobalAveragePooling2D()(input)
    se = Dense(num_channels // ratio, activation='relu')(se)
    se = Dense(num_channels, activation='sigmoid')(se)
    se = tf.reshape(se, [-1, 1, 1, num_channels])
    return Multiply()([input, se])

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

