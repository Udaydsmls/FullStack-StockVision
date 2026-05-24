from __future__ import annotations

import pandas as pd

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("prophet")
class ProphetModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="prophet",
        description="Facebook Prophet additive model; handles trend and seasonality.",
        paper="Taylor & Letham, 'Forecasting at Scale' (2017)",
        backend="prophet",
    )

    def build(self, window: int, num_features: int):
        raise NotImplementedError("Prophet does not use a Keras graph; call fit() instead.")

    def _frame_to_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"])
        else:
            dates = pd.to_datetime(df.index)
        return pd.DataFrame({"ds": dates, "y": df["Close"].astype(float).values})

    def fit(self, df: pd.DataFrame, *, horizon: int = 1):
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise RuntimeError("prophet is not installed: pip install prophet") from exc

        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(self._frame_to_prophet(df))
        return model

    def predict_next(self, fitted, df: pd.DataFrame) -> float:
        future = fitted.make_future_dataframe(periods=1, freq="B")
        forecast = fitted.predict(future.tail(1))
        return float(forecast["yhat"].iloc[-1])
