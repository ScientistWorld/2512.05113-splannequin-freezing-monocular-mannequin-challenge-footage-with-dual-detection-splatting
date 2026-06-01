from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


EXPERIMENT = "synthetic_validation_paper"
METRICS = ("psnr", "ssim", "lpips", "fvd", "inference_overhead")


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def ssim_global(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux, muy = pred.mean(), target.mean()
    vx, vy = pred.var(), target.var()
    cov = ((pred - mux) * (target - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))


def lpips_proxy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def fvd_proxy(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape[0] < 2:
        return 0.0
    return 1000.0 * float(np.mean(np.abs(np.diff(pred, axis=0) - np.diff(target, axis=0))))


def discover_prediction_files(checkpoints_dir: Path) -> dict[str, Path]:
    predictions = {}
    for path in sorted(checkpoints_dir.glob("*/predictions.json")):
        method_name = path.parent.name
        predictions[method_name] = path
    if not predictions:
        raise SystemExit(f"No method predictions found under {checkpoints_dir}/*/predictions.json")
    return predictions


def load_train_indices(num_views: int) -> np.ndarray:
    indices = np.arange(0, num_views, 2, dtype=np.int64)
    if indices.size == 0:
        raise SystemExit("Train split is empty; prediction tensors must contain at least one view")
    return indices


def evaluate_prediction(path: Path, target: np.ndarray, indices: np.ndarray) -> tuple[dict[str, float], float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pred = np.asarray(raw["pred_freeze"], dtype=np.float32)
    if pred.shape != target.shape:
        raise SystemExit(f"{path} has prediction shape {pred.shape}, expected {target.shape}")
    pred_slice = pred[indices]
    target_slice = target[indices]
    metrics = {
        "psnr": psnr(pred_slice, target_slice),
        "ssim": ssim_global(pred_slice, target_slice),
        "lpips": lpips_proxy(pred_slice, target_slice),
        "fvd": fvd_proxy(pred_slice, target_slice),
    }
    return metrics, float(raw["inference_state_values"])


def attach_inference_overhead(results: dict[str, dict[str, float]], state_values: dict[str, float]) -> None:
    baseline_key = "baseline" if "baseline" in state_values else sorted(state_values)[0]
    baseline_state = state_values[baseline_key]
    if baseline_state <= 0.0:
        raise SystemExit(f"Baseline inference_state_values for {baseline_key} must be positive")
    for name, metrics in results.items():
        metrics["inference_overhead"] = state_values[name] / baseline_state - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_mannequin.json")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--out", default="scoring/scores_train.json")
    args = parser.parse_args()

    scene = json.loads(Path(args.data).read_text(encoding="utf-8"))
    target = np.asarray(scene["gt_freeze"], dtype=np.float32)
    indices = load_train_indices(target.shape[0])
    prediction_files = discover_prediction_files(Path(args.checkpoints))

    results = {}
    state_values = {}
    for name, path in prediction_files.items():
        metrics, state_value = evaluate_prediction(path, target, indices)
        results[name] = {metric: metrics[metric] for metric in METRICS if metric in metrics}
        state_values[name] = state_value
    attach_inference_overhead(results, state_values)

    scores = {"experiments": {EXPERIMENT: {"results": results}}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
