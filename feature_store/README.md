# Feast feature store (optional)

A Feast project that turns the inline technical indicators into a managed
feature view. The inline pandas computation in
`stockvision/features.py` remains the default; Feast is opt-in via the
`STOCKVISION_USE_FEAST=1` environment variable.

## Layout

```
feature_store/
├── feature_repo/
│   ├── feature_store.yaml     # Local SQLite registry/online store (dev)
│   └── features.py            # Entity + FeatureView definitions
├── materialize.py             # Build a parquet from cached CSVs and apply()
└── feature_client.py          # get_features(ticker) helper
```

## Usage

```bash
pip install 'feast>=0.38.0'

# Apply the repo (writes the registry)
cd feature_store/feature_repo
feast apply

# Materialise indicators from cached CSVs
cd ..
python materialize.py --data-dir ../ingest_train/data --repo feature_repo

# Enable the Feast path in the feature builder
export STOCKVISION_USE_FEAST=1
```

When the flag is on and a ticker is provided, the most recent indicator
row from Feast is overlaid onto the last row of the inline frame. Production
deployments can swap the local SQLite registry/online store for Redis and S3
by editing `feature_repo/feature_store.yaml`.
