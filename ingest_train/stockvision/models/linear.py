"""Flattened linear regression: the baseline every other architecture has to beat."""

from .registry import register


def build(window, num_features):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, num_features), name="input"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, name="output"),
        ],
        name="linear",
    )


register("linear", "Flattened linear baseline.", build=build)
