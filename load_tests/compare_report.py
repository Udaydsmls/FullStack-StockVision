"""Turn Locust's stats CSV into a side-by-side Markdown table of the three backends."""

import argparse
import csv
from pathlib import Path

BACKENDS = {
    "fastapi:/predict": "Python FastAPI",
    "cpp:/predict": "C++ / ONNX Runtime",
    "triton:/predict": "Triton",
}

COLUMNS = ["Requests/s", "Median Response Time", "95%", "Failure Count"]
HEADERS = ["Backend", "Requests/s", "Median (ms)", "p95 (ms)", "Failures"]


def build_table(stats_path):
    with stats_path.open(newline="") as stats_file:
        rows = {row["Name"]: row for row in csv.DictReader(stats_file)}

    lines = ["| " + " | ".join(HEADERS) + " |", "| " + " | ".join(["---"] * len(HEADERS)) + " |"]
    for name, label in BACKENDS.items():
        row = rows.get(name, {})
        values = [row.get(column, "-") for column in COLUMNS]
        lines.append("| " + " | ".join([label] + values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("results/run_stats.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/comparison.md"))
    args = parser.parse_args()

    if not args.stats.exists():
        raise SystemExit(f"Stats file not found: {args.stats}. Run locust with --csv first.")

    table = build_table(args.stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"# Backend comparison\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
