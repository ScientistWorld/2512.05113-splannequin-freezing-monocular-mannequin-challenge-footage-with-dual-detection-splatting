# Evaluation

## Benchmark

Procedural scenes from `data/scenes.py`. A monocular camera moves forward down a hallway-like 2D scene (lateral position x, depth z) and records a 128-pixel RGB image line per frame (96 frames, with small hand-held sway). Walls and backdrop are static. Six groups of subjects make small, smooth, random unintended motions. Frames carry sensor noise (σ = 0.01). Images are formed by Gaussian splatting with fixed conventions: perspective projection with focal 0.78 and near plane 0.2, an isotropic footprint with 0.3 px² dilation, front-to-back compositing, alpha ≤ 0.99, contributions below 1/255 skipped, and termination once transmittance would fall below 1e-4.

Two experiments share the protocol:

- `synthetic_validation_paper` (weight 0.7): near-static subjects (the paper's setting).
- `synthetic_large_motion` (weight 0.3): about 2× larger and about 2× faster-decorrelating subject motion. This is a stress test of where near-static assumptions break down.

Visible data:

- `data/train/<experiment>/scene_*.json`: 6 public training scenes per experiment, with inputs and answers (true per-frame scene states and the freeze instants they are scored at).
- `data/test/<experiment>/*.json`: 16 test scenes per experiment, inputs only. The answers and the freeze instants are held by the hidden evaluator.

Inputs per scene are `frames` [96][128][3], camera poses `cam_x`/`cam_z`, normalized `timestamps`, a sparse noisy initial point cloud (`init_points`: one position and color per primitive), and `primitive_budget` (64).

## Submission Format

A method is a directory `checkpoints/<method>/` containing `predictions/<experiment>/<scene_id>.json` for every visible scene of both experiments (train and test):

```json
{"frozen_states": {
  "positions": [96][K][2],   // (x, z) of each Gaussian at each timestamp
  "scale":     [96][K],      // world-unit isotropic scale > 0
  "opacity":   [96][K],      // in [0, 1]
  "color":     [96][K][3]    // RGB in [0, 1]
}}
```

There must be one frozen scene state for every timestamp, with 1 ≤ K ≤ 64 primitives. The evaluator picks which instants to score; on test they are private. Every other field is ignored.

## Validity Gate

These checks enforce the zero-inference-overhead and fixed-capacity constraint. A method fails as a whole if any scene file is missing or unreadable, a shape is wrong, a value is non-finite or out of range, or K exceeds 64. A failed method receives the worst value of every metric in every experiment: PSNR 0, SSIM −1, LPIPS proxy 1, FVD proxy 2000, edge error 2, instant selectivity −1.

## Metrics

For each scene and each evaluated freeze instant t*, the evaluator renders the submitted state for t* from all 96 camera poses with its own rasterizer. It compares this freeze-time video with the true state at t* rendered the same way. Freeze instants are every 8th frame from a per-scene offset, 12 per scene. Values are averaged over instants and scenes.

- `psnr` ↑ (primary): mean per-view PSNR, capped monotonically at 100 dB.
- `ssim` ↑: windowed SSIM along the image line (Gaussian σ = 1.5 px).
- `lpips` ↓: perceptual-error proxy, the mean absolute error over a 4-level resolution pyramid.
- `fvd` ↓: video-distance proxy, 1000 × the mean absolute error of view-to-view temporal derivatives (flicker).
- `edge_mae` ↓: error of image-axis gradients (blur, ghosting, double contours).
- `instant_selectivity` ↑ (constraint): for each scene, Σ(|R_j − Ḡ| − |R_j − G_j|) / Σ|G_j − Ḡ| over the evaluated instants j, where Ḡ is the mean target over instants. It measures the fraction of the true instant-specific change that is recovered. The range is [−1, 1]: 1 for perfect renders, 0 for any method that renders the same scene at every instant. Pixels that never change do not count.

## How to Run

- `bash scripts/evaluate_train.sh` scores every method directory on disk on the public training scenes and writes `scoring/scores_train.json` plus a per-scene breakdown in `scoring/diagnostics_train.json`.
- `bash scripts/evaluate.sh` writes the same train-slice numbers to `scoring/scores.json`.
- The hidden test slice scores every method directory on disk in the same way on the 16 private test scenes per experiment. The ranked slot is `checkpoints/splannequin/`, reported as `4dgaussians_splannequin`. If that slot is missing, it receives worst values.

Directory names map to slots as follows: `splannequin` → `4dgaussians_splannequin` and `baseline` → `4dgaussians`; any other name is used unchanged. Evaluation never imports or runs method code.
