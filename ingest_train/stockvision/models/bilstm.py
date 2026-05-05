from __future__ import annotations

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("bilstm")
class BiLSTMModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="bilstm",
        description="Bidirectional LSTM stack; captures forward and backward temporal dependencies.",
    )

    def build(self, window: int, num_features: int):
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
