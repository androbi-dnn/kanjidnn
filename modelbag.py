import numpy as np
import os
import warnings

from keras.models import Sequential
from keras.optimizers import Adam 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger

from keras import initializers
from keras import backend
from keras import layers
from keras import models

class ModelBag():

    def mobile_net_64(self, 
                input_shape=None,
                depth_multiplier=1, # DepthwiseConv2D param
                dropout=1e-3,
                input_tensor=None,
                classes=1000):

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
    # Type / Stride Filter Shape Input Size
    # Conv / s2     3 × 3 × 3 × 32      224 × 224 × 3       3 × 3 × 1 × 32  64 x 64 x 1     S1!
    # Conv dw / s1  3 × 3 × 32 dw       112 × 112 × 32                      64 x 64 x 32
    # Conv / s1     1 × 1 × 32 × 64     112 × 112 × 32                      64 x 64 x 32
    # Conv dw / s2  3 × 3 × 64 dw       112 × 112 × 64                      64 x 64 x 64    S1!     
    # Conv / s1     1 × 1 × 64 × 128    56 × 56 × 64                        64 x 64 x 64                
    # Conv dw / s1  3 × 3 × 128 dw      56 × 56 × 128                       64 x 64 x 128
    # Conv / s1     1 × 1 × 128 × 128   56 × 56 × 128                       64 x 64 x 128
    # Conv dw / s2  3 × 3 × 128 dw      56 × 56 × 128                       64 x 64 x 128
    # Conv / s1     1 × 1 × 128 × 256   28 × 28 × 128                       32 x 32 x 128
    # Conv dw / s1  3 × 3 × 256 dw      28 × 28 × 256                       32 x 32 x 256
    # Conv / s1     1 × 1 × 256 × 256   28 × 28 × 256                       32 x 32 x 256
    # Conv dw / s2  3 × 3 × 256 dw      28 × 28 × 256                       32 x 32 x 256
    # Conv / s1     1 × 1 × 256 × 512   14 × 14 × 256                       16 x 16 x 256

    # 5×
    # Conv dw / s1  3 × 3 × 512 dw      14 × 14 × 512                       16 x 16 x 512
    # Conv / s1     1 × 1 × 512 × 512   14 × 14 × 512                       16 x 16 x 512

    # Conv dw / s2  3 × 3 × 512 dw      14 × 14 × 512                       16 x 16 x 512
    # Conv / s1     1 × 1 × 512 × 1024  7 × 7 × 512                         8 x 8 x 512
    # Conv dw / s2  3 × 3 × 1024 dw     7 × 7 × 1024                        8 x 8 x 1024
    # Conv / s1     1 × 1 × 1024 × 1024 7 × 7 × 1024                        8 x 8 x 1024
    # Avg Pool / s1 Pool 7 × 7          7 × 7 × 1024      8 × 8             8 x 8 x 1024
    # FC / s1       1024 × 1000        1 × 1 × 1024                         1 × 1 × 1024             
    # Softmax / s1  1 × 1 × 1000                          1 x 1 x NKanji

        init = initializers.TruncatedNormal(stddev=0.09)

        x = self._conv_block(img_input, 32, strides=(1, 1), kernel_initializer = init) # strides=(2, 2)
        x = self._depthwise_conv_block(x, 64, depth_multiplier, block_id=1, kernel_initializer = init)
    
        x = self._depthwise_conv_block(x, 128, depth_multiplier, strides=(1, 1), block_id=2, kernel_initializer = init) # strides=(2, 2)
        x = self._depthwise_conv_block(x, 128, depth_multiplier, block_id=3, kernel_initializer = init)

        x = self._depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4, kernel_initializer = init)
        x = self._depthwise_conv_block(x, 256, depth_multiplier, block_id=5, kernel_initializer = init)

        x = self._depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6, kernel_initializer = init)
        x = self._depthwise_conv_block(x, 512, depth_multiplier, block_id=7, kernel_initializer = init)
        x = self._depthwise_conv_block(x, 512, depth_multiplier, block_id=8, kernel_initializer = init)
        x = self._depthwise_conv_block(x, 512, depth_multiplier, block_id=9, kernel_initializer = init)
        x = self._depthwise_conv_block(x, 512, depth_multiplier, block_id=10, kernel_initializer = init)
        x = self._depthwise_conv_block(x, 512, depth_multiplier, block_id=11, kernel_initializer = init)

        x = self._depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12, kernel_initializer = init)
        x = self._depthwise_conv_block(x, 1024, depth_multiplier, block_id=13, kernel_initializer = init)


        if backend.image_data_format() == 'channels_first':
            shape = (1024, 1, 1)
        else:
            shape = (1, 1, 1024)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(dropout, name='dropout')(x)
        x = layers.Conv2D(classes, (1, 1),
                            padding='same',
                            kernel_initializer = init,
                            name='conv_preds')(x)
        x = layers.Activation('softmax', name='act_softmax')(x)
        x = layers.Reshape((classes,), name='reshape_2')(x)

        model = models.Model(img_input, x, name='mobilenet_64')
        return model

    def _conv_block(self, inputs, filters, kernel=(3, 3), strides=(1, 1), kernel_initializer = None):

        channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')(inputs) # ((0, 1), (0, 1))
        x = layers.Conv2D(filters, kernel,
                        padding='valid',
                        use_bias=False,
                        strides=strides,
                        kernel_initializer = kernel_initializer,
                        name='conv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
        return layers.ReLU(6., name='conv1_relu')(x)

    def _depthwise_conv_block(self, inputs, pointwise_conv_filters,
                            depth_multiplier=1, strides=(1, 1), block_id=1, kernel_initializer = None):
        """Adds a depthwise convolution block."""

        channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

        if strides == (1, 1):
            x = inputs
        else:
            x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                    name='conv_pad_%d' % block_id)(inputs)
        x = layers.DepthwiseConv2D((3, 3),
                                padding='same' if strides == (1, 1) else 'valid',
                                depth_multiplier=depth_multiplier,
                                strides=strides,
                                use_bias=False,
                                kernel_initializer = kernel_initializer,
                                name='conv_dw_%d' % block_id)(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
        x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

        x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                        padding='same',
                        use_bias=False,
                        strides=(1, 1),
                        kernel_initializer = kernel_initializer,
                        name='conv_pw_%d' % block_id)(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                    name='conv_pw_%d_bn' % block_id)(x)
        return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
 
    def vgg_m7_1_9(self, input_shape=None, classes=1000):
        """This is the second "M7_1" that appears in Charlie Tsai's 
        https://github.com/charlietsai/japanese-handwriting-nn/blob/master/models/vgg_models.py
        which corresponds to M9 in Table 1 of the paper.
        """
        model = Sequential( name='vgg_m7_1_9')

        model.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=input_shape, kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',
                        kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same',
                        kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same',
                        kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4096, activation="relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.5))

        model.add(Dense(4096, activation="relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.5))

        model.add(Dense(classes, activation="softmax"))

        return model

    def M16_drop(self, input_shape=None, classes=1000):

        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation="relu",
                        name='conv1_1', input_shape=input_shape, kernel_initializer='he_normal'))
        model.add(Conv2D(64, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv2_1'))
        model.add(Conv2D(128, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(256, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv3_1'))
        model.add(Conv2D(256, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv3_2'))
        model.add(Conv2D(256, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(512, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv4_1'))
        model.add(Conv2D(512, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv4_2'))
        model.add(Conv2D(512, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv5_1'))
        model.add(Conv2D(512, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv5_2'))
        model.add(Conv2D(512, (3, 3), padding="same",
                        activation='relu', kernel_initializer='he_normal', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(4096, activation="relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation="relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation="softmax"))

        return model
