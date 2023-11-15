import tensorflow as tf
from tensorflow import keras
from keras import layers

class ResidualBiGRU(keras.Model):
    def __init__(self, hidden_size, dropout_rate, n_layers=1):
        super().__init__()
        self.grus = keras.Sequential([layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True, dropout=dropout_rate))
                    for _ in range(n_layers)])
        self.fc1 = layers.TimeDistributed(layers.Dense(hidden_size * 4))
        self.ln1 = layers.LayerNormalization()
        self.fc2 = layers.TimeDistributed(layers.Dense(hidden_size))
        self.ln2 = layers.LayerNormalization()
    
    def call(self, inputs):
        res = inputs
        res = self.grus(res)
        res = self.ln1(self.fc1(res))
        res = keras.activations.relu(res)
        res = self.ln2(self.fc2(res))
        res = keras.activations.relu(res)

        out = res + inputs
        return out
    
class MultiResidualBiGRU(keras.Model):
    def __init__(self, hidden_size, out_size, n_layers, dropout_rate):
        super().__init__()
        self.fc_in = layers.Dense(hidden_size)
        self.ln = layers.LayerNormalization()
        self.res_bigrus = keras.Sequential([ResidualBiGRU(hidden_size, dropout_rate, n_layers) for _ in range(n_layers)])
        self. fc_out = layers.TimeDistributed(layers.Dense(out_size))

    def call(self, inputs):
        x = self.fc_in(inputs)
        x = self.ln(x)
        x = keras.activations.relu(x)

        x = self.res_bigrus(x)
        out = self.fc_out(x)
        return out

class EncoderLayer(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding):
        super().__init__()
        self.conv = layers.Conv1D(filters, kernel_size, strides, padding)
        self.ln = layers.LayerNormalization()
    def call(self, inputs):
        x = self.ln(self.conv(inputs))
        out = keras.activations.relu(x)
        return out
    
class GRUNet(keras.Model):
    def __init__(self, conv_arch, hidden_size, n_layers, output_num, gru_dropout_rate=0.2, dropout_rate=0.4):
        super().__init__()
        self.conv = keras.Sequential([EncoderLayer(filters, kernel_size, strides, padding='same')
                                    for _, filters, kernel_size, strides in conv_arch], name='conv')
        self.res_bigrus = keras.Sequential([ResidualBiGRU(hidden_size, gru_dropout_rate, n_layers) for _ in range(n_layers)])
        #MultiResidualBiGRU(hidden_size, hidden_size, n_layers, gru_dropout_rate)
        self.convtranspose = keras.Sequential(sum([[layers.Conv1DTranspose(filters, kernel_size, strides, padding='same'),
                                               layers.Conv1D(filters, kernel_size, strides=1, padding='same', activation='relu'),
                                               layers.Conv1D(filters, kernel_size, strides=1, padding='same', activation='relu')]
                                               for filters, _, kernel_size, strides in reversed(conv_arch)], []), name='convtrans')
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Conv1D(output_num, 1, 1, activation='sigmoid')
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.res_bigrus(x)
        x = self.convtranspose(x)
        x = self.dropout(x)
        outputs = self.output_layer(x)
        return outputs
    
if __name__ == '__main__':
    hidden_units = 64
    n_layers = 4
    arch = [(2, 8, 17, 2),
        (8, 32, 11, 2),
        (32, hidden_units, 7, 2)]
    model = GRUNet(arch, hidden_units, n_layers,3)
    model.build(input_shape=(None, 25000, 2))
    model.summary()