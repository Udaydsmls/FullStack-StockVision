"""Encoder-only Transformer (Vaswani et al., 2017) over the price window."""

from .registry import register

NUM_BLOCKS = 2
NUM_HEADS = 4
HEAD_SIZE = 32
FF_DIM = 64
DROPOUT = 0.1


def _encoder_block(x):
    """Self-attention then a feed-forward layer, each with a residual connection."""
    import tensorflow as tf

    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=HEAD_SIZE, dropout=DROPOUT
    )(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention)

    feed_forward = tf.keras.layers.Dense(FF_DIM, activation="relu")(x)
    feed_forward = tf.keras.layers.Dense(x.shape[-1])(feed_forward)
    feed_forward = tf.keras.layers.Dropout(DROPOUT)(feed_forward)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + feed_forward)


def build(window, num_features):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(window, num_features), name="input")
    x = tf.keras.layers.Dense(HEAD_SIZE * NUM_HEADS)(inputs)  # project into the model width
    for _ in range(NUM_BLOCKS):
        x = _encoder_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    outputs = tf.keras.layers.Dense(1, name="output")(x)
    return tf.keras.Model(inputs, outputs, name="transformer")


register("transformer", "Multi-head self-attention encoder.", build=build)
