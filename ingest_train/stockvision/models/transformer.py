from __future__ import annotations

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("transformer")
class TransformerModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="transformer",
        description="Multi-head self-attention encoder; benefits from longer windows and rich feature sets.",
        paper="Vaswani et al., 'Attention Is All You Need' (2017)",
    )

    num_blocks: int = 2
    num_heads: int = 4
    head_size: int = 32
    ff_dim: int = 64
    dropout: float = 0.1

    def _encoder_block(self, x):
        import tensorflow as tf

        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_size,
            dropout=self.dropout,
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = tf.keras.layers.Dense(self.ff_dim, activation="relu")(x)
        ff = tf.keras.layers.Dense(x.shape[-1])(ff)
        ff = tf.keras.layers.Dropout(self.dropout)(ff)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

    def build(self, window: int, num_features: int):
        import tensorflow as tf

        inputs = tf.keras.layers.Input(shape=(window, num_features), name="input")
        x = tf.keras.layers.Dense(self.head_size * self.num_heads)(inputs)
        for _ in range(self.num_blocks):
            x = self._encoder_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(1, name="output")(x)
        return tf.keras.Model(inputs, outputs, name="transformer")
