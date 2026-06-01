from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def temporal_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(abs(np.mean(np.abs(np.diff(pred, axis=0))) - np.mean(np.abs(np.diff(target, axis=0)))))


def ghosting_error(pred: np.ndarray, target: np.ndarray) -> float:
    # High-frequency residual along the image axis captures double contours/blur.
    pred_edge = np.diff(pred, axis=1)
    target_edge = np.diff(target, axis=1)
    return float(np.mean(np.abs(pred_edge - target_edge)))


def evaluate_one(path: Path, gt: np.ndarray) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pred = np.asarray(raw["pred_freeze"], dtype=np.float32)
    return {
        "freeze_psnr": psnr(pred, gt),
        "ghosting_error": ghosting_error(pred, gt),
        "temporal_error": temporal_error(pred, gt),
        "inference_state_values": float(raw["inference_state_values"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_mannequin.json")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out", default="scoring/scores.json")
    args = parser.parse_args()
    if len(args.inputs) != len(args.names):
        raise SystemExit("--inputs and --names must have the same length")
    scene = json.loads(Path(args.data).read_text(encoding="utf-8"))
    gt = np.asarray(scene["gt_freeze"], dtype=np.float32)
    results = {}
    for name, input_path in zip(args.names, args.inputs):
        results[name] = evaluate_one(Path(input_path), gt)
    scores = {"experiments": {"synthetic_freeze_time": {"results": results}}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
