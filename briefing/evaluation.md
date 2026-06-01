# Evaluation

## Metrics

The paper evaluates fixed-time reconstruction quality with full-reference metrics on synthetic data and no-reference perceptual metrics on real video.

For this reusable reproduction, the runnable benchmark uses a synthetic monocular forward-camera scene with known fixed-time ground truth:

- `psnr`: pixel fidelity between rendered fixed-time views and static ground truth. Higher is better.
- `ssim`: global structural agreement with the fixed-time target. Higher is better.
- `lpips`: lightweight perceptual-error proxy from RGB residuals. Lower is better.
- `fvd`: lightweight video-distance proxy from cross-view temporal derivative error. Lower is better.
- `foreground_mae`: non-background residual, included to discourage background-only shortcuts. Lower is better.
- `edge_mae`: high-frequency residual, included to penalize blur and double contours. Lower is better.
- `inference_overhead`: relative increase in rendering-time state size over the baseline. Lower is better and checks the zero-overhead claim.

## Baselines and Targets

The baseline is a dynamic reconstruction trained only on diagonal video observations and then rendered at a fixed timestamp. The proposed result should improve fixed-time quality without increasing inference state.

The paper reports synthetic improvements of its proposed system over its dynamic reconstruction baseline: PSNR 28.85 vs 28.03, SSIM 0.83 vs 0.81, LPIPS 0.08 vs 0.09, and FVD 82.73 vs 98.93.

## Evaluation Protocol

Train on the full monocular sequence, then render all input camera poses at a single selected timestamp. Evaluation reads saved render outputs from `checkpoints/*/predictions.json` and compares them to `data/synthetic_mannequin.json`. The evaluator does not import or call the method implementation.
