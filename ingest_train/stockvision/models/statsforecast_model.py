from __future__ import annotations

import pandas as pd

from .base import BaseForecastModel, ModelMetadata
from .registry import register_model


@register_model("autoarima")
class AutoArimaModel(BaseForecastModel):
    metadata = ModelMetadata(
        name="autoarima",
        description="Automatic ARIMA from Nixtla's statsforecast.",
        paper="Hyndman & Khandakar, 'Automatic Time Series Forecasting' (2008)",
        backend="statsforecast",
    )

    def build(self, window: int, num_features: int):
        raise NotImplementedError("AutoARIMA does not use a Keras graph; call fit() instead.")

    def _series(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"])
        else:
            dates = pd.to_datetime(df.index)
        return pd.DataFrame(
            {
                "unique_id": "ticker",
                "ds": dates,
                "y": df["Close"].astype(float).values,
            }
        )

    def fit(self, df: pd.DataFrame, *, horizon: int = 1):
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA
        except ImportError as exc:
            raise RuntimeError("statsforecast is not installed: pip install statsforecast") from exc

        sf = StatsForecast(models=[AutoARIMA(season_length=5)], freq="B", n_jobs=1)
        sf.fit(self._series(df))
        return sf

    def predict_next(self, fitted, df: pd.DataFrame) -> float:
        forecast = fitted.predict(h=1)
        col = "AutoARIMA" if "AutoARIMA" in forecast.columns else forecast.columns[-1]
        return float(forecast[col].iloc[-1])
