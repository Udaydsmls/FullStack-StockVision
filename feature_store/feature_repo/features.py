"""Feast feature definitions: entities and feature views for stock indicators."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

ticker = Entity(
    name="ticker",
    value_type=ValueType.STRING,
    description="Stock ticker symbol (e.g. AAPL).",
)

indicators_source = FileSource(
    name="stock_indicators_source",
    path="data/stock_indicators.parquet",
    event_timestamp_column="event_timestamp",
)

stock_indicators_fv = FeatureView(
    name="stock_indicators_fv",
    entities=[ticker],
    ttl=timedelta(days=30),
    schema=[
        Field(name="RSI_14", dtype=Float32),
        Field(name="MACD_12_26", dtype=Float32),
        Field(name="BB_upper_20", dtype=Float32),
        Field(name="BB_lower_20", dtype=Float32),
        Field(name="SMA_5", dtype=Float32),
        Field(name="SMA_20", dtype=Float32),
        Field(name="volume_ratio", dtype=Float32),
    ],
    online=True,
    source=indicators_source,
)
