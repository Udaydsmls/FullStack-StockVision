"""Build a Markdown comparison from Locust CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

BACKENDS = {
    "fastapi:/predict": "FastAPI",
    "cpp:/predict": "C++ server",
    "triton:/predict": "Triton (FastAPI shim)",
}


def _read_stats(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Name") or row.get("Endpoint") or ""
            if name in BACKENDS:
                rows[name] = row
    return rows


def render(stats: dict[str, dict[str, str]]) -> str:
    lines = [
        "| Backend | Requests/s | Median (ms) | p95 (ms) | Failures |",
        "| --- | --- | --- | --- | --- |",
    ]
    for endpoint, label in BACKENDS.items():
        row = stats.get(endpoint, {})
        if not row:
            lines.append(f"| {label} | - | - | - | - |")
            continue
        lines.append(
            "| {label} | {rps} | {median} | {p95} | {fail} |".format(
                label=label,
                rps=row.get("Requests/s", row.get("Current RPS", "-")),
                median=row.get("Median Response Time", "-"),
                p95=row.get("95%", row.get("95%ile", "-")),
                fail=row.get("Failure Count", row.get("Failures/s", "-")),
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("results/stats.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/comparison.md"))
    args = parser.parse_args(argv)

    if not args.stats.exists():
        print(f"Stats file not found: {args.stats}")
        return 1
    stats = _read_stats(args.stats)
    body = render(stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("# Load-test comparison\n\n" + body + "\n")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
