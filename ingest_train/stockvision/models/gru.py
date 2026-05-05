from __future__ import annotations

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("gru")
class GRUModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="gru",
        description="Stacked GRU regressor; faster to train than LSTM, comparable accuracy on shorter windows.",
    )

    def build(self, window: int, num_features: int):
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
