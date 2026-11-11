"""Procedural freeze-time benchmark scenes (public generator).

A monocular camera moves forward through a hallway-like 2D scene (lateral
position x, depth z) and records a 1D RGB image line per frame. The scene is
"almost frozen": walls and backdrop are static, but the subjects keep making
small, smooth, unintended motions (random smooth trajectories), as people do in
Mannequin-Challenge footage.

The task is to render the scene as it was at one chosen instant t* from every
camera pose of the video (a freeze-time video). The target for instant t* is
the true scene state at t* rendered from all camera poses.

Rendering follows 3D Gaussian splatting conventions in a 1D image: perspective
projection, an isotropic Gaussian footprint with a 0.3 px^2 low-pass dilation,
front-to-back alpha compositing, alpha clamped to 0.99, contributions below
1/255 skipped, and early termination once transmittance would drop below 1e-4.

This module is method-agnostic. It writes the public training scenes
(inputs plus answers) used by the visible train evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WIDTH = 128
NUM_FRAMES = 96
FOCAL = 0.78
NEAR = 0.2
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
TRANSMITTANCE_MIN = 1e-4
DILATION_PX2 = 0.3
BACKGROUND = 0.0
FREEZE_STRIDE = 8
NOISE_STD = 0.01
CAMERA_TRAVEL = 6.2

NUM_CLUSTERS = 6
PRIMS_PER_CLUSTER = 6
NUM_WALL = 20
NUM_BACKDROP = 8
NUM_PRIMITIVES = NUM_CLUSTERS * PRIMS_PER_CLUSTER + NUM_WALL + NUM_BACKDROP  # 64

# Two motion regimes: the paper's near-static setting, and the larger/faster
# subject motion the paper lists as a limitation.
REGIMES = {
    "synthetic_validation_paper": {"amplitude": (0.04, 0.08), "lengthscale": 0.22},
    "synthetic_large_motion": {"amplitude": (0.09, 0.16), "lengthscale": 0.12},
}

# Public training scenes. Test scenes use private seeds held by the evaluator.
TRAIN_SEEDS = {
    "synthetic_validation_paper": [101, 102, 103, 104, 105, 106],
    "synthetic_large_motion": [201, 202, 203, 204, 205, 206],
}


def render_views(
    positions: np.ndarray,
    scale: np.ndarray,
    opacity: np.ndarray,
    color: np.ndarray,
    cam_x: np.ndarray,
    cam_z: np.ndarray,
    width: int = WIDTH,
) -> np.ndarray:
    """Render one static Gaussian state from N cameras.

    positions [K,2] (x, z), scale [K] (world units), opacity [K] in [0,1],
    color [K,3] in [0,1], cam_x/cam_z [N]. Returns images [N, width, 3].
    """
    positions = np.asarray(positions, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    color = np.asarray(color, dtype=np.float64)
    cam_x = np.atleast_1d(np.asarray(cam_x, dtype=np.float64))
    cam_z = np.atleast_1d(np.asarray(cam_z, dtype=np.float64))
    u = np.arange(width, dtype=np.float64)[None, None, :]
    dz = positions[None, :, 1] - cam_z[:, None]  # [N,K]
    in_front = dz > NEAR
    dz_safe = np.where(in_front, dz, 1.0)
    px = (0.5 + FOCAL * (positions[None, :, 0] - cam_x[:, None]) / dz_safe) * (width - 1)
    sigma = np.sqrt((scale[None, :] * FOCAL * (width - 1) / dz_safe) ** 2 + DILATION_PX2)
    alpha = opacity[None, :, None] * np.exp(-0.5 * ((u - px[:, :, None]) / sigma[:, :, None]) ** 2)
    alpha = np.minimum(alpha, ALPHA_MAX)
    alpha = np.where((alpha >= ALPHA_MIN) & in_front[:, :, None], alpha, 0.0)
    order = np.argsort(np.where(in_front, dz, np.inf), axis=1, kind="stable")  # nearest first
    a = np.take_along_axis(alpha, order[:, :, None], axis=1)
    c = color[order]  # [N,K,3]
    t_after = np.cumprod(1.0 - a, axis=1)
    keep = t_after >= TRANSMITTANCE_MIN
    t_before = np.concatenate([np.ones_like(t_after[:, :1]), t_after[:, :-1]], axis=1)
    weight = a * t_before * keep
    rgb = np.einsum("nkw,nkc->nwc", weight, c)
    t_final = np.prod(np.where(keep, 1.0 - a, 1.0), axis=1)
    rgb = rgb + t_final[:, :, None] * BACKGROUND
    return np.clip(rgb, 0.0, 1.0)


def smooth_noise(rng: np.random.Generator, n_series: int, times: np.ndarray, lengthscale: float, n_features: int = 32) -> np.ndarray:
    """Unit-variance smooth random trajectories (random Fourier features of an RBF-kernel GP)."""
    omega = rng.normal(0.0, 1.0 / lengthscale, size=(n_series, n_features))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_series, n_features))
    vals = np.cos(omega[:, :, None] * times[None, None, :] + phase[:, :, None]).sum(axis=1)
    return np.sqrt(2.0 / n_features) * vals


def make_scene(seed: int, experiment: str, num_frames: int = NUM_FRAMES, width: int = WIDTH) -> tuple[dict, dict]:
    """Return (inputs, answers) for one scene."""
    regime = REGIMES[experiment]
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 1.0, num_frames)

    cam_z = CAMERA_TRAVEL * times
    cam_x = 0.03 * smooth_noise(rng, 1, times, 0.3)[0]

    # Subjects: clusters spread along the hallway, each a few primitives.
    centers_z = np.linspace(1.8, 8.2, NUM_CLUSTERS) + rng.uniform(-0.4, 0.4, NUM_CLUSTERS)
    centers_x = rng.uniform(-0.3, 0.3, NUM_CLUSTERS)
    cluster_of = np.repeat(np.arange(NUM_CLUSTERS), PRIMS_PER_CLUSTER)
    n_sub = cluster_of.size
    sub_pos = np.stack(
        [
            centers_x[cluster_of] + rng.uniform(-0.08, 0.08, n_sub),
            centers_z[cluster_of] + rng.uniform(-0.06, 0.06, n_sub),
        ],
        axis=1,
    )
    base_colors = rng.uniform(0.2, 0.95, size=(NUM_CLUSTERS, 3))
    sub_color = np.clip(base_colors[cluster_of] + rng.normal(0.0, 0.06, size=(n_sub, 3)), 0.02, 0.98)
    sub_opacity = rng.uniform(0.7, 0.98, n_sub)
    sub_scale = rng.uniform(0.025, 0.05, n_sub)

    # Static walls on both sides.
    side = np.where(np.arange(NUM_WALL) % 2 == 0, -1.0, 1.0)
    wall_pos = np.stack([side * rng.uniform(0.5, 0.75, NUM_WALL), rng.uniform(1.0, 9.0, NUM_WALL)], axis=1)
    grey = rng.uniform(0.35, 0.75, NUM_WALL)
    wall_color = np.clip(
        np.stack([grey, 0.92 * grey, 0.8 * grey], axis=1) + rng.normal(0.0, 0.04, size=(NUM_WALL, 3)), 0.02, 0.98
    )
    wall_opacity = rng.uniform(0.5, 0.9, NUM_WALL)
    wall_scale = rng.uniform(0.04, 0.08, NUM_WALL)

    # Static backdrop at the end of the hallway.
    back_pos = np.stack(
        [np.linspace(-1.8, 1.8, NUM_BACKDROP) + rng.uniform(-0.1, 0.1, NUM_BACKDROP), rng.uniform(9.3, 9.8, NUM_BACKDROP)],
        axis=1,
    )
    back_color = rng.uniform(0.1, 0.45, size=(NUM_BACKDROP, 3))
    back_opacity = rng.uniform(0.85, 0.95, NUM_BACKDROP)
    back_scale = rng.uniform(0.25, 0.4, NUM_BACKDROP)

    base_pos = np.concatenate([sub_pos, wall_pos, back_pos], axis=0)
    color = np.concatenate([sub_color, wall_color, back_color], axis=0)
    opacity = np.concatenate([sub_opacity, wall_opacity, back_opacity], axis=0)
    scale = np.concatenate([sub_scale, wall_scale, back_scale], axis=0)
    k = base_pos.shape[0]

    # Unintended subject motion: shared per-cluster sway plus per-primitive jitter.
    amp = rng.uniform(regime["amplitude"][0], regime["amplitude"][1], NUM_CLUSTERS)
    ls = regime["lengthscale"]
    cdx = amp[:, None] * smooth_noise(rng, NUM_CLUSTERS, times, ls)
    cdz = 0.5 * amp[:, None] * smooth_noise(rng, NUM_CLUSTERS, times, ls)
    pdx = 0.35 * amp[cluster_of][:, None] * smooth_noise(rng, n_sub, times, ls)
    pdz = 0.175 * amp[cluster_of][:, None] * smooth_noise(rng, n_sub, times, ls)
    motion = np.zeros((num_frames, k, 2))
    motion[:, :n_sub, 0] = (cdx[cluster_of] + pdx).T
    motion[:, :n_sub, 1] = (cdz[cluster_of] + pdz).T
    states = base_pos[None, :, :] + motion

    states = np.round(states, 6)
    color = np.round(color, 6)
    opacity = np.round(opacity, 6)
    scale = np.round(scale, 6)
    cam_x = np.round(cam_x, 6)
    cam_z = np.round(cam_z, 6)

    frames = np.stack(
        [render_views(states[t], scale, opacity, color, cam_x[t : t + 1], cam_z[t : t + 1], width)[0] for t in range(num_frames)]
    )
    frames = np.clip(frames + rng.normal(0.0, NOISE_STD, size=frames.shape), 0.0, 1.0)

    offset = int(rng.integers(0, FREEZE_STRIDE))
    freeze_indices = list(range(offset, num_frames, FREEZE_STRIDE))

    # A sparse, noisy initial point cloud (one point per primitive, shuffled),
    # as structure-from-motion would provide for a near-static video.
    perm = rng.permutation(k)
    init_pos = states.mean(axis=0) + rng.normal(0.0, 1.0, size=(k, 2)) * np.array([0.03, 0.06])
    init_col = np.clip(color + rng.normal(0.0, 0.05, size=(k, 3)), 0.02, 0.98)

    scene_id = f"{experiment}_{seed}"
    inputs = {
        "scene_id": scene_id,
        "experiment": experiment,
        "width": width,
        "num_frames": num_frames,
        "focal": FOCAL,
        "near": NEAR,
        "timestamps": np.round(times, 8).tolist(),
        "cam_x": cam_x.tolist(),
        "cam_z": cam_z.tolist(),
        "frames": np.round(frames, 4).tolist(),
        "init_points": {
            "positions": np.round(init_pos[perm], 6).tolist(),
            "colors": np.round(init_col[perm], 6).tolist(),
        },
        "primitive_budget": int(k),
    }
    answers = {
        "scene_id": scene_id,
        "experiment": experiment,
        "positions": states.tolist(),
        "scale": scale.tolist(),
        "opacity": opacity.tolist(),
        "color": color.tolist(),
        "cam_x": cam_x.tolist(),
        "cam_z": cam_z.tolist(),
        "freeze_indices": freeze_indices,
    }
    return inputs, answers


def write_train_scenes(out_root: Path, force: bool = False) -> None:
    for experiment, seeds in TRAIN_SEEDS.items():
        out_dir = out_root / experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, seed in enumerate(seeds):
            path = out_dir / f"scene_{idx:02d}.json"
            if path.exists() and not force:
                continue
            inputs, answers = make_scene(seed, experiment)
            inputs["answers"] = answers
            path.write_text(json.dumps(inputs), encoding="utf-8")
            print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/train")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_train_scenes(Path(args.out_root), force=args.force)


if __name__ == "__main__":
    main()
