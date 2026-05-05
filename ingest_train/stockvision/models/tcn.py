from __future__ import annotations

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("tcn")
class TCNModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="tcn",
        description="Dilated causal convolutions with residual connections; large receptive field at low compute cost.",
        paper="Bai, Kolter, Koltun, 'An Empirical Evaluation of Generic Convolutional and Recurrent Networks' (2018)",
    )

    filters: int = 32
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1

    def _residual_block(self, x, dilation: int):
        import tensorflow as tf

        prev = x
        for _ in range(2):
            x = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
            )(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        if prev.shape[-1] != self.filters:
            prev = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1, padding="same")(prev)
        return tf.keras.layers.Add()([prev, x])

    def build(self, window: int, num_features: int):
        import tensorflow as tf

        inputs = tf.keras.layers.Input(shape=(window, num_features), name="input")
        x = inputs
        for dilation in self.dilations:
            x = self._residual_block(x, dilation)
        x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, name="output")(x)
        return tf.keras.Model(inputs, outputs, name="tcn")
