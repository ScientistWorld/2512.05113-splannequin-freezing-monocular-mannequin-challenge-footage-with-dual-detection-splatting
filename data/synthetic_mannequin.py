"""Synthetic forward-camera mannequin benchmark.

The generator is method-agnostic: it defines a small dynamic Gaussian
scene, observed along a forward camera path, plus fixed-time ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _project(points: np.ndarray, cam_z: float, width: int) -> tuple[np.ndarray, np.ndarray]:
    dz = points[:, 1] - cam_z
    u = 0.5 + 0.78 * points[:, 0] / np.maximum(dz, 1e-3)
    px = u * (width - 1)
    return px, dz


def render_gaussians(
    positions: np.ndarray,
    colors: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    cam_z: float,
    width: int = 96,
    background: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a 1D RGB Gaussian splat image and return visibility/contribution."""
    x = np.arange(width, dtype=np.float32)[None, :]
    px, dz = _project(positions, cam_z, width)
    visible = (dz > 0.18) & (px > -0.15 * width) & (px < 1.15 * width)
    sigma = np.clip(scale * width / np.maximum(dz, 0.35), 0.8, 7.5)[:, None]
    depth_order = np.argsort(-dz)
    trans = np.ones((width,), dtype=np.float32)
    rgb = np.full((width, 3), background, dtype=np.float32)
    contrib = np.zeros((positions.shape[0],), dtype=np.float32)
    for k in depth_order:
        if not visible[k]:
            continue
        weight = opacity[k] * np.exp(-0.5 * ((x[0] - px[k]) / sigma[k, 0]) ** 2)
        alpha = np.clip(weight, 0.0, 0.97) * trans
        contrib[k] = float(alpha.sum())
        rgb += alpha[:, None] * colors[k]
        trans *= 1.0 - alpha
    return np.clip(rgb, 0.0, 1.0), visible.astype(bool), contrib


def create_scene(seed: int = 3, num_points: int = 34, num_frames: int = 24, width: int = 96) -> dict:
    rng = np.random.default_rng(seed)
    z = np.linspace(1.0, 7.2, num_points)
    x = rng.uniform(-0.55, 0.55, size=num_points) + 0.12 * np.sin(2.4 * z)
    base_pos = np.stack([x, z], axis=1).astype(np.float32)
    colors = rng.uniform(0.15, 0.95, size=(num_points, 3)).astype(np.float32)
    opacity = rng.uniform(0.22, 0.62, size=num_points).astype(np.float32)
    scale = rng.uniform(0.018, 0.04, size=num_points).astype(np.float32)
    moving = rng.choice(num_points, size=max(6, num_points // 4), replace=False)
    amp = np.zeros(num_points, dtype=np.float32)
    amp[moving] = rng.uniform(0.035, 0.105, size=len(moving))
    phase = rng.uniform(0.0, 2 * np.pi, size=num_points).astype(np.float32)
    times = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    cam_z = np.linspace(-0.15, 5.6, num_frames, dtype=np.float32)
    states = []
    for t in times:
        pos = base_pos.copy()
        pos[:, 0] += amp * np.sin(2 * np.pi * (1.15 * t) + phase)
        pos[:, 1] += 0.018 * amp * np.cos(2 * np.pi * t + phase)
        states.append(pos)
    states = np.stack(states, axis=0).astype(np.float32)

    frames, masks, contribs = [], [], []
    for i in range(num_frames):
        img, vis, con = render_gaussians(states[i], colors, opacity, scale, float(cam_z[i]), width)
        frames.append(img)
        masks.append(vis)
        contribs.append(con)

    freeze_index = num_frames // 2
    gt_freeze = []
    for i in range(num_frames):
        img, _, _ = render_gaussians(states[freeze_index], colors, opacity, scale, float(cam_z[i]), width)
        gt_freeze.append(img)

    return {
        "seed": seed,
        "width": width,
        "num_frames": num_frames,
        "freeze_index": freeze_index,
        "cam_z": cam_z.tolist(),
        "frames": np.stack(frames).tolist(),
        "gt_freeze": np.stack(gt_freeze).tolist(),
        "base_positions": base_pos.tolist(),
        "true_states": states.tolist(),
        "colors": colors.tolist(),
        "opacity": opacity.tolist(),
        "scale": scale.tolist(),
        "diagonal_visible": np.stack(masks).astype(int).tolist(),
        "diagonal_contrib": np.stack(contribs).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_mannequin.json")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--points", type=int, default=34)
    parser.add_argument("--width", type=int, default=96)
    args = parser.parse_args()
    scene = create_scene(args.seed, args.points, args.frames, args.width)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene), encoding="utf-8")


if __name__ == "__main__":
    main()
