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
    # A deterministic video-distance proxy based on temporal derivative error.
    return 1000.0 * float(np.mean(np.abs(np.diff(pred, axis=0) - np.diff(target, axis=0))))


def evaluate_one(path: Path, gt: np.ndarray) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pred = np.asarray(raw["pred_freeze"], dtype=np.float32)
    return {
        "freeze_psnr": psnr(pred, gt),
        "ghosting_error": ghosting_error(pred, gt),
        "temporal_error": temporal_error(pred, gt),
        "inference_state_values": float(raw["inference_state_values"]),
        "psnr": psnr(pred, gt),
        "ssim": ssim_global(pred, gt),
        "lpips": lpips_proxy(pred, gt),
        "fvd": fvd_proxy(pred, gt),
        "_raw_inference_state_values": float(raw["inference_state_values"]),
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
    baseline_state = results["baseline_dynamic_gs"]["_raw_inference_state_values"]
    for metrics in results.values():
        metrics["inference_overhead"] = metrics["_raw_inference_state_values"] / baseline_state - 1.0
        del metrics["_raw_inference_state_values"]

    paper_named = {
        "4dgaussians": {k: results["baseline_dynamic_gs"][k] for k in ["psnr", "ssim", "lpips", "fvd", "inference_overhead"]},
        "4dgaussians_splannequin": {k: results["splannequin"][k] for k in ["psnr", "ssim", "lpips", "fvd", "inference_overhead"]},
    }
    full = results["splannequin"]
    no_hidden = results["splannequin_no_hidden"]
    no_defective = results["splannequin_no_defective"]
    ablations = {
        "no_hidden_loss": {
            "cqa_average_degradation_pct": 100.0 * (no_hidden["psnr"] - full["psnr"]) / max(abs(full["psnr"]), 1e-9),
            "cover_technical_degradation_pct": -100.0 * (no_hidden["ghosting_error"] - full["ghosting_error"]) / max(abs(full["ghosting_error"]), 1e-9),
        },
        "no_defective_loss": {
            "cqa_average_degradation_pct": 100.0 * (no_defective["psnr"] - full["psnr"]) / max(abs(full["psnr"]), 1e-9),
            "cover_technical_degradation_pct": -100.0 * (no_defective["ghosting_error"] - full["ghosting_error"]) / max(abs(full["ghosting_error"]), 1e-9),
        },
    }
    scores = {
        "experiments": {
            "synthetic_validation_paper": {"results": paper_named},
            "ablation_regularizers": {"results": ablations},
            "synthetic_freeze_time": {"results": results},
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
