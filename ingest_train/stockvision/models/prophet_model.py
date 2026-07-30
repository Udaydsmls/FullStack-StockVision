"""Prophet: an additive trend + seasonality model (Taylor & Letham, 2017).

Prophet is not a neural network, so there is nothing to export to ONNX. It
trains on the raw close prices and is stored with joblib instead.
"""

import pandas as pd

from .registry import register


def _to_prophet_frame(df):
    """Prophet wants exactly two columns: ds (date) and y (value)."""
    dates = pd.to_datetime(df["Date"] if "Date" in df.columns else df.index)
    return pd.DataFrame({"ds": dates, "y": df["Close"].astype(float).values})


def fit(df):
    from prophet import Prophet

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(_to_prophet_frame(df))
    return model


def predict_next(fitted, df):
    # "B" = business days, so the next row is the next trading day.
    future = fitted.make_future_dataframe(periods=1, freq="B")
    return float(fitted.predict(future.tail(1))["yhat"].iloc[-1])


register(
    "prophet",
    "Additive trend and seasonality model.",
    backend="prophet",
    fit=fit,
    predict_next=predict_next,
)
