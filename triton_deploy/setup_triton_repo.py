"""Copy the exported ONNX models into the folder layout Triton expects."""

import argparse
import json
import shutil
from pathlib import Path

CONFIG_TEMPLATE = """name: "{name}"
backend: "onnxruntime"
max_batch_size: 32
input [
  {{
    name: "{input_name}"
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


def build_repository(artifacts_dir, output_dir):
    """Walk artifacts/<TICKER>/<MODEL>/ and write output/<ticker>_<model>/."""
    deployed = []
    for metadata_path in sorted(artifacts_dir.glob("*/*/metadata.json")):
        model_dir = metadata_path.parent
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            continue  # Prophet and AutoARIMA have no ONNX graph to serve

        metadata = json.loads(metadata_path.read_text())
        name = f"{model_dir.parent.name}_{model_dir.name}".lower()

        version_dir = output_dir / name / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, version_dir / "model.onnx")
        (output_dir / name / "config.pbtxt").write_text(
            CONFIG_TEMPLATE.format(
                name=name,
                window=metadata["window"],
                num_features=metadata["num_features"],
                input_name=metadata["input_name"],
                output_name=metadata["output_name"],
            )
        )
        deployed.append(name)
    return deployed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("../ingest_train/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("model_repository"))
    args = parser.parse_args()

    if not args.artifacts.exists():
        raise SystemExit(f"Artifacts directory not found: {args.artifacts}")

    deployed = build_repository(args.artifacts, args.output)
    if not deployed:
        raise SystemExit("No ONNX models found. Run `stockvision train` first.")

    print(f"Deployed {len(deployed)} models to {args.output}:")
    for name in deployed:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
