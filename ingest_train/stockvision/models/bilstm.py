"""Bidirectional LSTM: reads each window forwards and backwards."""

from .registry import register


def build(window, num_features):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, num_features), name="input"),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48, return_sequences=True)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, name="output"),
        ],
        name="bilstm",
    )


register("bilstm", "Bidirectional LSTM stack.", build=build)
