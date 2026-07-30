"""Stacked GRU: faster to train than the LSTM, similar accuracy on short windows."""

from .registry import register


def build(window, num_features):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, num_features), name="input"),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, name="output"),
        ],
        name="gru",
    )


register("gru", "Stacked GRU; faster than LSTM with similar accuracy.", build=build)
