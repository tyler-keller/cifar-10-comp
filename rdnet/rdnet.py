'''
rewrite of https://github.com/naver-ai/rdnet/blob/main/rdnet/rdnet.py in tensorflow
'''

import tensorflow as tf
from tensorflow import keras
from keras import layers

growth_k = 24
nb_block = 2
init_learning_rate = 1e-4
epsilon = 1e-4
dropout_rate = 0.2

nesterov_momentum = 0.9
weight_decay = 1e-4

class_num = 10

def conv_layer(input, filters, kernel_size, strides=1, layer_name="conv"):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=layer_name)(input)

def global_average_pooling(x):
    return layers.GlobalAveragePooling2D()(x)

def batch_normalization(x, training, name):
    return layers.BatchNormalization(momentum=0.9, epsilon=epsilon, name=name)(x, training=training)

def dropout(x, rate, training):
    return layers.Dropout(rate)(x, training=training)

def relu(x):
    return layers.Activation('relu')(x)

def average_pooling(x, pool_size=(2, 2), strides=2, padding='valid'):
    return layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(x)

def max_pooling(x, pool_size=(3, 3), strides=2, padding='valid'):
    return layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(x)

def concatenation(layers_list):
    return layers.Concatenate(axis=-1)(layers_list)

def linear(x, units):
    return layers.Dense(units, name='linear')(x)

# DenseNet class
class DenseNet:
    def __init__(self, input_shape, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.build_network(input_shape)

    def bottleneck_layer(self, x, scope):
        x = batch_normalization(x, training=self.training, name=f'{scope}_batch1')
        x = relu(x)
        x = conv_layer(x, filters=4 * self.filters, kernel_size=1, layer_name=f'{scope}_conv1')
        x = dropout(x, rate=dropout_rate, training=self.training)

        x = batch_normalization(x, training=self.training, name=f'{scope}_batch2')
        x = relu(x)
        x = conv_layer(x, filters=self.filters, kernel_size=3, layer_name=f'{scope}_conv2')
        x = dropout(x, rate=dropout_rate, training=self.training)
        return x

    def transition_layer(self, x, scope):
        x = batch_normalization(x, training=self.training, name=f'{scope}_batch1')
        x = relu(x)
        x = conv_layer(x, filters=int(x.shape[-1]) // 2, kernel_size=1, layer_name=f'{scope}_conv1')
        x = dropout(x, rate=dropout_rate, training=self.training)
        x = average_pooling(x)
        return x

    def dense_block(self, x, nb_layers, layer_name):
        layers_concat = [x]
        for i in range(nb_layers):
            bottleneck_output = self.bottleneck_layer(x, scope=f'{layer_name}_bottleN_{i}')
            layers_concat.append(bottleneck_output)
            x = concatenation(layers_concat)
        return x

    def build_network(self, input_shape):
        inputs = keras.Input(shape=input_shape)

        x = conv_layer(inputs, filters=2 * self.filters, kernel_size=7, strides=2, layer_name='conv0')
        x = max_pooling(x, pool_size=(3, 3), strides=2, padding='same')

        for i in range(self.nb_blocks):
            x = self.dense_block(x, nb_layers=6, layer_name=f'dense_{i + 1}')
            if i != self.nb_blocks - 1:
                x = self.transition_layer(x, scope=f'trans_{i + 1}')

        x = self.dense_block(x, nb_layers=32, layer_name='dense_final')

        x = batch_normalization(x, training=self.training, name='linear_batch')
        x = relu(x)
        x = global_average_pooling(x)
        outputs = layers.Dense(units=class_num, activation='softmax')(x)
        return keras.Model(inputs, outputs)