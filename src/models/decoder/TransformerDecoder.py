import tensorflow as tf
from keras import layers
import os
from tensorflow import keras
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_nlp

class TransformerDecoder(keras.Model):
    def __init__(
        self,
        n_classes: int,
        intermediate_dim: int,
        num_heads: int,
        n_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.decoder = keras.Sequential([keras_nlp.layers.TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads, dropout=dropout)
                        for _ in range(n_layers)])
        self.fc = layers.TimeDistributed(layers.Dense(n_classes, activation='sigmoid'))
    def call(self, inputs):
        x = self.decoder(inputs)
        outputs = self.fc(x)
        return outputs
