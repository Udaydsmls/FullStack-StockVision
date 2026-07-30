"""AutoARIMA from Nixtla's statsforecast (Hyndman & Khandakar, 2008).

Like Prophet this is a classical model, so it is pickled rather than exported
to ONNX and is only served by the FastAPI backend.
"""

import pandas as pd

from .registry import register


def _to_statsforecast_frame(df):
    """statsforecast wants one row per (series id, date, value)."""
    dates = pd.to_datetime(df["Date"] if "Date" in df.columns else df.index)
    return pd.DataFrame(
        {"unique_id": "ticker", "ds": dates, "y": df["Close"].astype(float).values}
    )


def fit(df):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    # season_length=5 because a trading week is five days long.
    model = StatsForecast(models=[AutoARIMA(season_length=5)], freq="B", n_jobs=1)
    model.fit(_to_statsforecast_frame(df))
    return model


def predict_next(fitted, df):
    forecast = fitted.predict(h=1)
    return float(forecast["AutoARIMA"].iloc[-1])


register(
    "autoarima",
    "Automatic ARIMA order selection.",
    backend="statsforecast",
    fit=fit,
    predict_next=predict_next,
)
