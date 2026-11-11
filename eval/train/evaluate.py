"""Visible train-slice evaluator for freeze-time reconstruction.

Scores every method directory on disk (``<checkpoints>/<method>/predictions/``)
on the public training scenes in ``data/train`` (whose answers are public).
A submission is a frozen Gaussian scene state for every timestamp of every
scene; this evaluator renders each freeze instant listed in the scene answers
from every camera pose with its own renderer and compares against the true
freeze-time video. Nothing a method writes besides those states is read.
The hidden test slice uses the same interface on private scenes.

Failure handling: a method whose outputs are missing, malformed, non-finite,
out of range, or over the primitive budget fails as a whole and receives the
worst value of every metric in every experiment. Required slots that are not
on disk are scored the same way.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SLICE = "train"
EXPERIMENTS = ("synthetic_validation_paper", "synthetic_large_motion")
# Slot names the ranking expects; scored with worst values when absent.
REQUIRED_SLOTS = ("4dgaussians_splannequin",)
SLOT_ALIASES = {"splannequin": "4dgaussians_splannequin", "baseline": "4dgaussians"}
PRIMITIVE_BUDGET = 64
METRICS = ("psnr", "ssim", "lpips", "fvd", "edge_mae", "instant_selectivity")
WORST = {
    "psnr": 0.0,
    "ssim": -1.0,
    "lpips": 1.0,
    "fvd": 2000.0,
    "edge_mae": 2.0,
    "instant_selectivity": -1.0,
}
PSNR_CAP = 100.0

# Renderer constants (fixed by the benchmark).
FOCAL = 0.78
NEAR = 0.2
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
TRANSMITTANCE_MIN = 1e-4
DILATION_PX2 = 0.3
BACKGROUND = 0.0
WIDTH = 128


class SubmissionError(Exception):
    pass


def render_views(positions, scale, opacity, color, cam_x, cam_z, width=WIDTH):
    positions = np.asarray(positions, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    color = np.asarray(color, dtype=np.float64)
    cam_x = np.atleast_1d(np.asarray(cam_x, dtype=np.float64))
    cam_z = np.atleast_1d(np.asarray(cam_z, dtype=np.float64))
    u = np.arange(width, dtype=np.float64)[None, None, :]
    dz = positions[None, :, 1] - cam_z[:, None]
    in_front = dz > NEAR
    dz_safe = np.where(in_front, dz, 1.0)
    px = (0.5 + FOCAL * (positions[None, :, 0] - cam_x[:, None]) / dz_safe) * (width - 1)
    sigma = np.sqrt((scale[None, :] * FOCAL * (width - 1) / dz_safe) ** 2 + DILATION_PX2)
    alpha = opacity[None, :, None] * np.exp(-0.5 * ((u - px[:, :, None]) / sigma[:, :, None]) ** 2)
    alpha = np.minimum(alpha, ALPHA_MAX)
    alpha = np.where((alpha >= ALPHA_MIN) & in_front[:, :, None], alpha, 0.0)
    order = np.argsort(np.where(in_front, dz, np.inf), axis=1, kind="stable")
    a = np.take_along_axis(alpha, order[:, :, None], axis=1)
    c = color[order]
    t_after = np.cumprod(1.0 - a, axis=1)
    keep = t_after >= TRANSMITTANCE_MIN
    t_before = np.concatenate([np.ones_like(t_after[:, :1]), t_after[:, :-1]], axis=1)
    weight = a * t_before * keep
    rgb = np.einsum("nkw,nkc->nwc", weight, c)
    t_final = np.prod(np.where(keep, 1.0 - a, 1.0), axis=1)
    rgb = rgb + t_final[:, :, None] * BACKGROUND
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------- metrics


def psnr(pred, gt):
    # Mean over views of per-image PSNR, monotone cap at PSNR_CAP (MSE floor 1e-10).
    mse = np.mean((pred - gt) ** 2, axis=(1, 2))
    return float(np.mean(np.minimum(PSNR_CAP, -10.0 * np.log10(np.maximum(mse, 1e-10)))))


def _gauss_window(sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()


_WINDOW = _gauss_window()


def _filter_rows(x):
    # x [N, W, C]; valid Gaussian filtering along the image axis.
    r = (len(_WINDOW) - 1) // 2
    n = x.shape[1] - 2 * r
    out = np.zeros((x.shape[0], n, x.shape[2]), dtype=np.float64)
    for j, wj in enumerate(_WINDOW):
        out += wj * x[:, j : j + n, :]
    return out


def ssim(pred, gt):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = _filter_rows(pred), _filter_rows(gt)
    sxx = _filter_rows(pred * pred) - mx * mx
    syy = _filter_rows(gt * gt) - my * my
    sxy = _filter_rows(pred * gt) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(np.mean(s))


def lpips_proxy(pred, gt):
    vals = []
    p, g = pred, gt
    for _ in range(4):
        vals.append(float(np.mean(np.abs(p - g))))
        w = (p.shape[1] // 2) * 2
        if w < 2:
            break
        p = 0.5 * (p[:, 0:w:2] + p[:, 1:w:2])
        g = 0.5 * (g[:, 0:w:2] + g[:, 1:w:2])
    return float(np.mean(vals))


def fvd_proxy(pred, gt):
    return 1000.0 * float(np.mean(np.abs(np.diff(pred, axis=0) - np.diff(gt, axis=0))))


def edge_mae(pred, gt):
    return float(np.mean(np.abs(np.diff(pred, axis=1) - np.diff(gt, axis=1))))


def instant_selectivity(renders, targets):
    """Fraction of the instant-specific appearance change that a reconstruction recovers.

    renders/targets [F, N, W, C] for the F evaluated freeze instants of one scene.
    sum(|R_j - mean_j G| - |R_j - G_j|) / sum(|G_j - mean_j G|), in [-1, 1]:
    1 for the true freeze-time videos, 0 for any reconstruction that renders the
    same image for every instant (no timestamp control), and pixels that do not
    change between instants contribute nothing.
    """
    mean_t = targets.mean(axis=0, keepdims=True)
    denom = float(np.abs(targets - mean_t).sum())
    if denom <= 1e-12:
        return 0.0
    gain = np.abs(renders - mean_t) - np.abs(renders - targets)
    return float(gain.sum() / denom)


# ---------------------------------------------------------------- scenes


def load_answer_scenes(root: Path) -> dict:
    scenes = {}
    for experiment in EXPERIMENTS:
        paths = sorted((root / experiment).glob("*.json"))
        if not paths:
            raise RuntimeError(f"no training scenes for {experiment} under {root}")
        items = []
        for path in paths:
            scene_raw = json.loads(path.read_text(encoding="utf-8"))
            raw = scene_raw["answers"]
            positions = np.asarray(raw["positions"], dtype=np.float64)
            cam_x = np.asarray(raw["cam_x"], dtype=np.float64)
            cam_z = np.asarray(raw["cam_z"], dtype=np.float64)
            freeze = [int(i) for i in raw["freeze_indices"]]
            scale = np.asarray(raw["scale"], dtype=np.float64)
            opacity = np.asarray(raw["opacity"], dtype=np.float64)
            color = np.asarray(raw["color"], dtype=np.float64)
            targets = np.stack([render_views(positions[f], scale, opacity, color, cam_x, cam_z) for f in freeze])
            items.append(
                {
                    "scene_id": raw["scene_id"],
                    "num_frames": int(positions.shape[0]),
                    "cam_x": cam_x,
                    "cam_z": cam_z,
                    "freeze": freeze,
                    "targets": targets,
                }
            )
        scenes[experiment] = items
    return scenes


# ---------------------------------------------------------------- submissions


def _array(raw, key, path):
    if key not in raw:
        raise SubmissionError(f"{path}: missing '{key}'")
    try:
        arr = np.asarray(raw[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SubmissionError(f"{path}: '{key}' is not a numeric array ({exc})") from exc
    if not np.all(np.isfinite(arr)):
        raise SubmissionError(f"{path}: '{key}' contains non-finite values")
    return arr


def load_frozen_states(path: Path, num_frames: int):
    if not path.is_file():
        raise SubmissionError(f"missing prediction file {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SubmissionError(f"{path}: unreadable JSON ({exc})") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("frozen_states"), dict):
        raise SubmissionError(f"{path}: expected an object with 'frozen_states'")
    fs = raw["frozen_states"]
    pos = _array(fs, "positions", path)
    scale = _array(fs, "scale", path)
    opacity = _array(fs, "opacity", path)
    color = _array(fs, "color", path)
    if pos.ndim != 3 or pos.shape[0] != num_frames or pos.shape[2] != 2:
        raise SubmissionError(f"{path}: positions must have shape [{num_frames}, K, 2], got {pos.shape}")
    k = pos.shape[1]
    if k < 1 or k > PRIMITIVE_BUDGET:
        raise SubmissionError(f"{path}: {k} primitives; the budget is 1..{PRIMITIVE_BUDGET}")
    if scale.shape != (num_frames, k) or opacity.shape != (num_frames, k) or color.shape != (num_frames, k, 3):
        raise SubmissionError(f"{path}: scale/opacity/color shapes do not match positions {pos.shape}")
    if np.any(opacity < 0.0) or np.any(opacity > 1.0):
        raise SubmissionError(f"{path}: opacity outside [0, 1]")
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise SubmissionError(f"{path}: color outside [0, 1]")
    if np.any(scale <= 0.0) or np.any(scale > 10.0):
        raise SubmissionError(f"{path}: scale outside (0, 10]")
    if np.any(np.abs(pos) > 1000.0):
        raise SubmissionError(f"{path}: position magnitude above 1000")
    return pos, scale, opacity, color


def evaluate_experiment(method_dir: Path, experiment: str, scenes: list, details: list | None) -> dict:
    per_video = {m: [] for m in ("psnr", "ssim", "lpips", "fvd", "edge_mae")}
    motion = []
    for scene in scenes:
        path = method_dir / "predictions" / experiment / f"{scene['scene_id']}.json"
        pos, scale, opacity, color = load_frozen_states(path, scene["num_frames"])
        renders = np.stack(
            [render_views(pos[f], scale[f], opacity[f], color[f], scene["cam_x"], scene["cam_z"]) for f in scene["freeze"]]
        )
        scene_vals = {m: [] for m in per_video}
        for r, g in zip(renders, scene["targets"]):
            scene_vals["psnr"].append(psnr(r, g))
            scene_vals["ssim"].append(ssim(r, g))
            scene_vals["lpips"].append(lpips_proxy(r, g))
            scene_vals["fvd"].append(fvd_proxy(r, g))
            scene_vals["edge_mae"].append(edge_mae(r, g))
        sel = instant_selectivity(renders, scene["targets"])
        motion.append(sel)
        for m in per_video:
            per_video[m].extend(scene_vals[m])
        if details is not None:
            details.append(
                {
                    "experiment": experiment,
                    "scene_id": scene["scene_id"],
                    "per_freeze_psnr": scene_vals["psnr"],
                    **{m: float(np.mean(v)) for m, v in scene_vals.items()},
                    "instant_selectivity": sel,
                }
            )
    out = {m: float(np.mean(v)) for m, v in per_video.items()}
    out["instant_selectivity"] = float(np.mean(motion))
    return out


def discover_methods(checkpoints: Path) -> dict:
    found = {}
    if checkpoints.is_dir():
        for d in sorted(checkpoints.iterdir()):
            if d.is_dir() and ((d / "predictions").exists() or (d / "predictions.json").exists()):
                found.setdefault(SLOT_ALIASES.get(d.name, d.name), []).append(d)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--scenes-root", default=str(ROOT / "data" / "train"))
    parser.add_argument("--out", default="scoring/scores_train.json")
    parser.add_argument("--details", default="scoring/diagnostics_train.json", help="per-scene breakdown")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        out.unlink()
    scenes = load_answer_scenes(Path(args.scenes_root))
    methods = discover_methods(Path(args.checkpoints))
    slots = sorted(set(methods) | set(REQUIRED_SLOTS))
    results = {exp: {} for exp in EXPERIMENTS}
    failures = {}
    details = {} if args.details else None
    for slot in slots:
        dirs = methods.get(slot, [])
        slot_details = [] if details is not None else None
        try:
            if not dirs:
                raise SubmissionError("no predictions on disk for this slot")
            if len(dirs) > 1:
                raise SubmissionError(f"ambiguous: {[d.name for d in dirs]} map to the same slot")
            per_exp = {exp: evaluate_experiment(dirs[0], exp, scenes[exp], slot_details) for exp in EXPERIMENTS}
            for exp, vals in per_exp.items():
                if not all(math.isfinite(v) for v in vals.values()):
                    raise SubmissionError(f"non-finite metric in {exp}")
        except Exception as exc:  # any failure scores the whole run as failed
            per_exp = {exp: dict(WORST) for exp in EXPERIMENTS}
            failures[slot] = f"{type(exc).__name__}: {exc}"
            print(f"[evaluate_train] {slot}: FAILED -> worst values ({failures[slot]})")
        for exp in EXPERIMENTS:
            results[exp][slot] = {m: per_exp[exp][m] for m in METRICS}
        if details is not None:
            details[slot] = slot_details
    scores = {
        "metadata": {
            "slice": SLICE,
            "scenes_per_experiment": {exp: len(scenes[exp]) for exp in EXPERIMENTS},
            "failed_methods": failures,
        },
        "experiments": {exp: {"results": results[exp]} for exp in EXPERIMENTS},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    if args.details:
        Path(args.details).write_text(json.dumps(details, indent=1), encoding="utf-8")
    print(f"[evaluate_train] wrote {out} for {len(slots)} method slot(s)")


if __name__ == "__main__":
    main()
