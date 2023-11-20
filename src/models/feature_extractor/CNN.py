from typing import Callable, Optional
from tensorflow import keras
from keras import layers

class CNN(keras.Model):
    def __init__(
        self,
        base_filters: tuple = (128,),
        kernel_sizes: tuple = (32, 16, 4, 2),
        strides: tuple = (1, 1, 1, 1),
        pooling: bool = False
    ):
        super().__init__()
        self.base_filters=base_filters
        self.kernel_sizes=kernel_sizes
        self.strides=strides
        self.pooling=pooling

        self.conv_blocks = []
        for kernel_size, stride in zip(kernel_sizes, strides):
            tmp_block = [
                layers.Conv1D(base_filters[0], kernel_size, strides=stride, padding='same')
            ]
            for filters in base_filters[1:]:
                tmp_block = tmp_block + [
                    layers.BatchNormalization(axis=1),
                    layers.ReLU(),
                    layers.Conv1D(filters, kernel_size, stride, 'same')
                ]
            self.conv_blocks.append(keras.Sequential(tmp_block))
        self.pooling = pooling
        if pooling:
            self.pooling_layer = layers.AveragePooling1D(pool_size=2, strides=2)

    def call(self, inputs):
        x = inputs
        for conv in self.conv_blocks:
            x = conv(x)
        if self.pooling:
            x = self.pooling_layer(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"base_filters": self.base_filters,
                       'kernel_sizes': self.kernel_sizes,
                       'strides':self.strides,
                       'pooling': self.pooling})
        return config