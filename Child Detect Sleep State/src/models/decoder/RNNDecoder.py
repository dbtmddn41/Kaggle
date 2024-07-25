import tensorflow as tf
from tensorflow import keras
from keras import layers

class LSTMDecoder(keras.Model):
    def __init__(
        self,
        n_classes: int,
        hidden_size: int,
        n_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstms = keras.Sequential([layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True, dropout=dropout))
                    for _ in range(n_layers)])
        self.fc = layers.TimeDistributed(layers.Dense(n_classes, activation='sigmoid'))
    def call(self, inputs):
        x = self.lstms(inputs)
        outputs = self.fc(x)
        return outputs


class GRUDecoder(keras.Model):
    def __init__(
        self,
        n_classes: int,
        hidden_size: int,
        n_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstms = keras.Sequential([layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True, dropout=dropout))
                    for _ in range(n_layers)])
        self.fc = layers.TimeDistributed(layers.Dense(n_classes, activation='sigmoid'))
    def call(self, inputs):
        x = self.lstms(inputs)
        outputs = self.fc(x)
        return outputs
