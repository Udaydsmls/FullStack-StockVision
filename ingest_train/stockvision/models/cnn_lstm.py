"""1-D convolutions pick up short patterns, then an LSTM handles the longer trend."""

from .registry import register


def build(window, num_features):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, num_features), name="input"),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.LSTM(48),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, name="output"),
        ],
        name="cnn_lstm",
    )


register("cnn_lstm", "1-D CNN front-end feeding an LSTM.", build=build)
