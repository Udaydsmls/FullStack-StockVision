"""Temporal Convolutional Network (Bai, Kolter & Koltun, 2018).

Stacked dilated causal convolutions see far back in time without the cost of
an RNN. Each dilation doubles, so four blocks cover 30 days of history.
"""

from .registry import register

FILTERS = 32
KERNEL_SIZE = 3
DILATIONS = (1, 2, 4, 8)
DROPOUT = 0.1


def _residual_block(x, dilation):
    """Two dilated convolutions, added back onto the block's input."""
    import tensorflow as tf

    shortcut = x
    for _ in range(2):
        x = tf.keras.layers.Conv1D(
            FILTERS,
            kernel_size=KERNEL_SIZE,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
        )(x)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
    if shortcut.shape[-1] != FILTERS:
        # Match the channel count so the two branches can be added.
        shortcut = tf.keras.layers.Conv1D(FILTERS, kernel_size=1, padding="same")(shortcut)
    return tf.keras.layers.Add()([shortcut, x])


def build(window, num_features):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(window, num_features), name="input")
    x = inputs
    for dilation in DILATIONS:
        x = _residual_block(x, dilation)
    x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)  # keep the most recent step
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, name="output")(x)
    return tf.keras.Model(inputs, outputs, name="tcn")


register("tcn", "Dilated causal convolutions with residual connections.", build=build)
