from typing import Callable, Optional
import tensorflow as tf
from tensorflow import keras
from keras import layers

class UNetDecoder(keras.Model):
    def __init__(
        self,
        n_classes: int,
        scale_factor: int = 2,
        se: bool = False,
        res: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.inc = DoubleConv(64)
        self.down1 = Down(128, scale_factor, se, res)
        self.down2 = Down(256, scale_factor, se, res)
        self.down3 = Down(512, scale_factor, se, res)
        self.down4 = Down(1024, scale_factor, se, res)
        self.up1 = layers.Conv1DTranspose(512, kernel_size=scale_factor, strides=scale_factor)
        self.conv1 = DoubleConv(512)
        self.up2 = layers.Conv1DTranspose(256, scale_factor, scale_factor)
        self.conv2 = DoubleConv(256)
        self.up3 = layers.Conv1DTranspose(128, scale_factor, scale_factor)
        self.conv3 = DoubleConv(128)
        self.up4 = layers.Conv1DTranspose(64, scale_factor, scale_factor)
        self.conv4 = DoubleConv(64)
        self.concat = layers.Concatenate(axis=-1)

        self.cls = keras.Sequential([
            layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            layers.Dropout(dropout),
            layers.Conv1D(n_classes, kernel_size=1, activation='sigmoid'),
        ]
        )

    def call(self, inputs):
        # 1D U-Net
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x)
        x = self.conv1(self.concat([x4, x]))
        x = self.up2(x)
        x = self.conv2(self.concat([x3, x]))
        x = self.up3(x)
        x = self.conv3(self.concat([x2, x]))
        x = self.up4(x)
        x = self.conv4(self.concat([x1, x]))
        outputs = self.cls(x)
    
        return outputs

class DoubleConv(keras.Model):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        out_channels,
        mid_channels=None,
        se=False,
        res=False,
    ):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        if se:
            non_linearity = SEModule(out_channels)
        else:
            non_linearity = layers.ReLU()
        self.double_conv = keras.Sequential([
            layers.Conv1D(mid_channels, kernel_size=3, padding='same', use_bias=False),
            layers.LayerNormalization(),
            layers.ReLU(),
            layers.Conv1D(out_channels, kernel_size=3, padding='same', use_bias=False),
            layers.LayerNormalization(),
            non_linearity,
        ]
        )

    def call(self, inputs):
        if self.res:
            return inputs + self.double_conv(inputs)
        else:
            return self.double_conv(inputs)

class SEModule(keras.Model):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = layers.AveragePooling1D()
        self.fc = keras.Sequential([
            layers.Dense(channel // reduction, use_bias=False, activation='relu'),
            layers.Dense(channel, use_bias=False, activation='sigmoid'),
        ]
        )

    def call(self, inputs):
        y = tf.squeeze(self.avg_pool(inputs))
        y = self.fc(y)[:, :, tf.newaxis]
        return inputs * y
    
class Down(keras.Model):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, out_channels, scale_factor, se=False, res=False
    ):
        super().__init__()
        self.maxpool_conv = keras.Sequential([
            layers.MaxPool1D(scale_factor, scale_factor),
            DoubleConv(out_channels, se=se, res=res),
        ]
        )

    def call(self, inputs):
        return self.maxpool_conv(inputs)
    
class Up(keras.Model):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, bilinear=True, scale_factor=2
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = layers.UpSampling2D((scale_factor, 1), interpolation='bilinear')#scale_factor=scale_factor, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(out_channels, in_channels // 2)
        else:
            self.up = layers.Conv1DTranspose(in_channels // 2, kernel_size=scale_factor, strides=scale_factor)
            self.conv = DoubleConv(out_channels)
        self.norm = layers.LayerNormalization()
    def call(self, inputs):
        x = tf.squeeze(self.up(inputs))
        x = self.conv(x)
        x = self.norm(x)
        return x