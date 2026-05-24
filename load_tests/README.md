# Load tests

Locust suite that exercises the three serving backends in parallel.

```bash
pip install 'locust>=2.24.0'

# Run against all three (FastAPI on 8000, C++ on 8080, Triton via FastAPI proxy)
locust -f locustfile.py --headless -u 100 -r 10 --run-time 60s \
       --csv results/run

# Build the comparison table
python compare_report.py --stats results/run_stats.csv --out results/comparison.md
```

Configure target URLs and tickers via environment variables:

| Variable                 | Default                  |
| ------------------------ | ------------------------ |
| `LOADTEST_FASTAPI`       | `http://localhost:8000`  |
| `LOADTEST_CPP`           | `http://localhost:8080`  |
| `LOADTEST_TRITON_PROXY`  | `http://localhost:8000`  |
| `LOADTEST_TICKERS`       | `AAPL,MSFT,TSLA,NVDA,GOOG` |
| `LOADTEST_MODELS`        | `lstm,gru,transformer`   |

All server code is untouched; this directory is external to every backend.
