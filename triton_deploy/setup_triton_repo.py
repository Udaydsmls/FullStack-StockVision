"""Lay out the Triton model_repository from ONNX files in ingest_train/artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


CONFIG_TEMPLATE = """name: "{name}"
backend: "onnxruntime"
max_batch_size: 32
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ {window}, {num_features} ]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
"""


def _read_metadata(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)


def build_repo(artifacts: Path, output: Path) -> list[str]:
    if not artifacts.exists():
        raise SystemExit(f"Artifacts directory not found: {artifacts}")
    output.mkdir(parents=True, exist_ok=True)

    deployed: list[str] = []
    for ticker_dir in sorted(artifacts.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for model_dir in sorted(ticker_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            onnx_path = model_dir / "model.onnx"
            meta_path = model_dir / "metadata.json"
            if not onnx_path.exists() or not meta_path.exists():
                continue
            metadata = _read_metadata(meta_path)
            name = f"{ticker_dir.name}_{model_dir.name}".lower()
            window = int(metadata["window"])
            num_features = int(metadata["num_features"])
            output_name = "output"

            target_dir = output / name / "1"
            _symlink_or_copy(onnx_path.resolve(), target_dir / "model.onnx")
            (output / name / "config.pbtxt").write_text(
                CONFIG_TEMPLATE.format(
                    name=name, window=window, num_features=num_features, output_name=output_name
                )
            )
            deployed.append(name)
    return deployed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("../ingest_train/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("model_repository"))
    args = parser.parse_args(argv)

    deployed = build_repo(args.artifacts, args.output)
    if not deployed:
        print("No ONNX models found. Run `stockvision train` first.", file=sys.stderr)
        return 1
    print("Deployed:")
    for name in deployed:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
