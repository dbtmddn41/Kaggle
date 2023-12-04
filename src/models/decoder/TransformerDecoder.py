import tensorflow as tf 
from tensorflow import keras
from keras import layers

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=self.dropout_rate)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation='relu'), layers.Dense(embed_dim)])
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(query=inputs, value=inputs, attention_mask=mask)
        x = inputs + self.dropout1(attention_output)
        x = self.layernorm_1(x)
        proj_output = self.dense_proj(x)
        x = self.dropout2(proj_output) + x
        output = self.layernorm_2(x)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "dense_dim": self.dense_dim, "num_heads": self.num_heads, 'dropout_rate': self.dropout_rate})
        return config


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def build(self, input_shape):
    super().build(input_shape)

  def call(self, x):
    input_shape = tf.shape(x)
    batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
    pos_input = tf.tile(tf.expand_dims(tf.keras.backend.arange(0, seq_len), axis = 0), [batch_size, 1])
    pos_input = tf.keras.backend.cast(pos_input, tf.float32)
    evens = tf.keras.backend.arange(0, output_dim // 2) * 2
    odds = tf.keras.backend.arange(0, output_dim // 2) * 2 + 1
    even_embedding = tf.sin(
      tf.keras.backend.dot(
        tf.expand_dims(pos_input, -1),
        tf.expand_dims(1.0 / tf.pow(
          10000.0,
          tf.cast(evens, dtype = tf.float32) / tf.cast(output_dim, dtype = tf.float32)
          ), 0)
        )
      )
    odd_embedding = tf.cos(
      tf.keras.backend.dot(
        tf.expand_dims(pos_input, -1),
        tf.expand_dims(1.0 / tf.pow(
          10000.0,
          tf.cast((odds - 1), dtype = tf.float32) / tf.cast(output_dim, dtype = tf.float32)
          ), 0)
        )
      )
    embedding = tf.stack([even_embedding, odd_embedding], axis = -1)
    output = tf.reshape(embedding, [-1, tf.shape(x)[1], output_dim])
    output += x

    return output
    
class TransformerEncoder(layers.Layer):
    def __init__(self, n_classes, num_blocks, embed_dim, dense_dim, num_heads, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        # self.sequence_len = sequence_len
        # self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.pos_embedding = PositionalEmbedding()
        block_layers = []
        for i in range(num_blocks):
            block_layers.append(TransformerEncoderBlock(embed_dim, dense_dim, num_heads, dropout_rate))
        self.block_layers = keras.Sequential(block_layers)
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.TimeDistributed(layers.Dense(n_classes, activation='sigmoid'))

    def call(self, inputs):
        pos_inputs = self.pos_embedding(inputs)
        encoded = self.block_layers(pos_inputs)
        outputs = self.fc(self.dropout(encoded))
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_blocks': self.num_blocks,
            'embed_dim': self.embed_dim, 'dropout_rate': self.dropout_rate,
            'dense_dim': self.dense_dim, 'num_heads': self.num_heads,
        })
        return config