from __future__ import annotations

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("lstm")
class LSTMModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="lstm",
        description="Two-layer LSTM with dropout, suited to medium-term sequence learning.",
    )

    def build(self, window: int, num_features: int):
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
