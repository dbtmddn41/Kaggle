import tensorflow as tf 
from tensorflow import keras
from keras import layers
from transformers import TFAutoModel

class DoubleConv(keras.Model):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        out_channels,
        mid_channels=None,
        res=False,
    ):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = keras.Sequential([
            layers.Conv1D(mid_channels, kernel_size=3, padding='same', use_bias=False),
            layers.LayerNormalization(),
            layers.ReLU(),
            layers.Conv1D(out_channels, kernel_size=3, padding='same', use_bias=False),
            layers.LayerNormalization(),
            layers.ReLU(),
        ]
        )

    def call(self, inputs):
        if self.res:
            return inputs + self.double_conv(inputs)
        else:
            return self.double_conv(inputs)

class Down(keras.Model):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, out_channels, scale_factor, res=False
    ):
        super().__init__()
        self.maxpool_conv = keras.Sequential([
            layers.MaxPool1D(scale_factor, scale_factor),
            DoubleConv(out_channels, res=res),
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
    

class HGTransformerDecoder(layers.Layer):
    def __init__(self, model_name, hidden_dim, down_nums, dropout_rate, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.down_nums = down_nums
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes

        model = TFAutoModel.from_pretrained(model_name)
        self.decoder = model.layers[0].encoder
        self.downsample_blocks = []
        self.upsample_blocks = []
        print(down_nums)
        for i in range(down_nums-1,-1,-1):
          self.downsample_blocks.append(Down(hidden_dim//(2**i), scale_factor=2))
          self.upsample_blocks.append(Up(hidden_dim//(4**(down_nums-i-1)), hidden_dim//(4**(down_nums-i)), bilinear=False, scale_factor=2))
        print(self.downsample_blocks)
        self.downsample_blocks = keras.Sequential(self.downsample_blocks)     
        self.upsample_blocks = keras.Sequential(self.upsample_blocks)     
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.TimeDistributed(layers.Dense(n_classes, activation='sigmoid'))

    def call(self, inputs):
        x = self.downsample_blocks(inputs)
        x = self.decoder(x, None)
        x = self.upsample_blocks(x)
        outputs = self.fc(self.dropout(x))
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'hidden_dim': self.hidden_dim, 'down_nums': self.down_nums,
            'dropout_rate': self.dropout_rate, 'n_classes': self.n_classes,
        })
        return config
