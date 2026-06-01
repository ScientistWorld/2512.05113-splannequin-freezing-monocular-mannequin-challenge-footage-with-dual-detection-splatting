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


def foreground_mae(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.max(target, axis=-1) > 0.04
    if not np.any(mask):
        return lpips_proxy(pred, target)
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def edge_mae(pred: np.ndarray, target: np.ndarray) -> float:
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
        "psnr": psnr(pred, gt),
        "ssim": ssim_global(pred, gt),
        "lpips": lpips_proxy(pred, gt),
        "fvd": fvd_proxy(pred, gt),
        "foreground_mae": foreground_mae(pred, gt),
        "edge_mae": edge_mae(pred, gt),
        "_raw_inference_state_values": float(raw["inference_state_values"]),
    }


PAPER_NAME_ALIASES = {
    "baseline": "4dgaussians",
    "baseline_dynamic_gs": "4dgaussians",
    "splannequin": "4dgaussians_splannequin",
}


def evaluate_group(data_path: Path, inputs: list[Path], names: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    scene = json.loads(data_path.read_text(encoding="utf-8"))
    gt = np.asarray(scene["gt_freeze"], dtype=np.float32)
    results = {}
    for name, input_path in zip(names, inputs):
        results[name] = evaluate_one(input_path, gt)
    baseline_key = "baseline_dynamic_gs" if "baseline_dynamic_gs" in results else names[0]
    baseline_state = results[baseline_key]["_raw_inference_state_values"]
    for metrics in results.values():
        metrics["inference_overhead"] = metrics["_raw_inference_state_values"] / baseline_state - 1.0
        del metrics["_raw_inference_state_values"]

    paper_named = {}
    for name, metrics in results.items():
        if name.startswith("splannequin_no_"):
            continue
        paper_name = PAPER_NAME_ALIASES.get(name, name)
        paper_named[paper_name] = {
            k: metrics[k]
            for k in ["psnr", "ssim", "lpips", "fvd", "foreground_mae", "edge_mae", "inference_overhead"]
        }
    return results, paper_named


def summarize_groups(groups: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    if not groups:
        return {}
    names = sorted(set.intersection(*(set(group) for group in groups)))
    candidate_metrics = [
        "psnr",
        "ssim",
        "lpips",
        "fvd",
        "foreground_mae",
        "edge_mae",
        "inference_overhead",
    ]
    summary = {}
    for name in names:
        metrics = [
            metric
            for metric in candidate_metrics
            if all(metric in group[name] for group in groups)
        ]
        summary[name] = {
            metric: float(np.mean([group[name][metric] for group in groups]))
            for metric in metrics
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_mannequin.json")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out", default="scoring/scores.json")
    args = parser.parse_args()
    if len(args.inputs) != len(args.names):
        raise SystemExit("--inputs and --names must have the same length")
    results, paper_named = evaluate_group(Path(args.data), [Path(p) for p in args.inputs], args.names)
    ablations = {}
    if {"splannequin", "splannequin_no_hidden", "splannequin_no_defective"}.issubset(results):
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
    extra_groups = []
    for scene_dir in sorted(Path("checkpoints").glob("extra_seed_*")):
        data_path = Path("data") / f"{scene_dir.name}.json"
        paths = [
            scene_dir / "baseline" / "predictions.json",
            scene_dir / "splannequin" / "predictions.json",
            scene_dir / "no_hidden" / "predictions.json",
            scene_dir / "no_defective" / "predictions.json",
        ]
        if data_path.exists() and all(path.exists() for path in paths):
            _, extra_paper_named = evaluate_group(
                data_path,
                paths,
                ["baseline_dynamic_gs", "splannequin", "splannequin_no_hidden", "splannequin_no_defective"],
            )
            extra_groups.append(extra_paper_named)

    diagnostics = {
        "experiments": {
            "synthetic_freeze_time": {"results": results},
        }
    }
    if ablations:
        diagnostics["experiments"]["ablation_regularizers"] = {"results": ablations}
    if extra_groups:
        diagnostics["experiments"]["synthetic_validation_extra_scenes"] = {"results": summarize_groups(extra_groups)}
        diagnostics["experiments"]["synthetic_validation_scored_aggregate"] = {
            "results": summarize_groups([paper_named, *extra_groups])
        }
    Path("scoring/diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    scored_results = summarize_groups([paper_named, *extra_groups]) if extra_groups else paper_named
    scores = {
        "experiments": {
            "synthetic_validation_paper": {"results": scored_results}
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
