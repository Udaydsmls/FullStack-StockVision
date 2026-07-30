# Load tests

One Locust run drives all three backends at the same time, so the throughput
numbers are directly comparable.

```bash
pip install 'locust>=2.24.0'

locust -f locustfile.py --headless -u 100 -r 10 --run-time 60s --csv results/run
python compare_report.py --stats results/run_stats.csv
```

`compare_report.py` writes `results/comparison.md` with a row per backend:
requests/s, median latency, p95 and failure count.

| Variable            | Default                    |
| ------------------- | -------------------------- |
| `LOADTEST_FASTAPI`  | `http://localhost:8000`    |
| `LOADTEST_CPP`      | `http://localhost:8080`    |
| `LOADTEST_TRITON`   | `http://localhost:8000`    |
| `LOADTEST_TICKERS`  | `AAPL,MSFT,TSLA,NVDA,GOOG` |
| `LOADTEST_MODELS`   | `lstm,gru,transformer`     |

Nothing in this folder is imported by the servers; it only talks to them over
HTTP.
