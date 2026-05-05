from __future__ import annotations

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("linear")
class LinearModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="linear",
        description="Flattened linear baseline; useful sanity check and competitive on short horizons.",
    )

    def build(self, window: int, num_features: int):
        import tensorflow as tf

        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(window, num_features), name="input"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, name="output"),
            ],
            name="linear",
        )
