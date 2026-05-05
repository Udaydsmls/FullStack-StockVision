from __future__ import annotations

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("cnn_lstm")
class CNNLSTMModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="cnn_lstm",
        description="1-D convolutional feature extractor feeding an LSTM; captures short-range patterns then long-range structure.",
    )

    def build(self, window: int, num_features: int):
        import tensorflow as tf

        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(window, num_features), name="input"),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"),
                tf.keras.layers.MaxPool1D(pool_size=2),
                tf.keras.layers.LSTM(48),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, name="output"),
            ],
            name="cnn_lstm",
        )
