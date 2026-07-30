"""Two-layer LSTM with dropout."""

from .registry import register


def build(window, num_features):
    # TensorFlow is imported here so that just listing the models stays fast.
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, num_features), name="input"),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, name="output"),
        ],
        name="lstm",
    )


register("lstm", "Two-layer LSTM with dropout.", build=build)
