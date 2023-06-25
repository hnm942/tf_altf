import tensorflow as tf
from tensorflow import keras
from keras import layers


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, padding="same", activation="relu"
        )
        # self.lstm1 = tf.keras.layers.LSTM(num_hid, return_sequences=True, activation="relu")
        # self.lstm2 = tf.keras.layers.LSTM(num_hid, return_sequences=True, activation="relu")
        # self.lstm3 = tf.keras.layers.LSTM(num_hid, return_sequences=True, activation="relu")
        # self.dense1 = tf.keras.layers.Dense(num_hid, activation="relu")
        # self.dense2 = tf.keras.layers.Dense(num_hid, activation="relu")
        # self.dense3 = tf.keras.layers.Dense(num_hid, activation="relu")

    def call(self, x):
        #x = x[..., tf.newaxis]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.dense1(x)
        # x = self.dense2(x)
        # x = self.dense3(x)
        return x