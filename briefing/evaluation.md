# Evaluation

## Metrics

The paper evaluates fixed-time reconstruction quality with full-reference metrics on synthetic data and no-reference perceptual metrics on real video.

For this reusable reproduction, the runnable benchmark uses a synthetic monocular forward-camera scene with known fixed-time ground truth:

- `freeze_psnr`: PSNR between rendered fixed-time views and static ground truth. Higher is better.
- `ghosting_error`: mean high-frequency residual between rendered and ground-truth views. Lower is better.
- `temporal_error`: mismatch in frame-to-frame variation along the fixed-time camera path. Lower is better.
- `inference_state_values`: number of scalar state values needed at rendering time. Lower is better and checks the zero-overhead claim.

## Baselines and Targets

The baseline is the same dynamic Gaussian representation trained only on diagonal video observations and then rendered at a fixed timestamp. The proposed result should improve fixed-time quality without increasing inference state.

The paper reports synthetic improvements for 4DGaussians + Splannequin over 4DGaussians: PSNR 28.85 vs 28.03, SSIM 0.83 vs 0.81, LPIPS 0.08 vs 0.09, and FVD 82.73 vs 98.93.

## Evaluation Protocol

Train on the full monocular sequence, then render all input camera poses at a single selected timestamp. Evaluation reads saved render outputs from `checkpoints/*/predictions.json` and compares them to `data/synthetic_mannequin.json`. The evaluator does not import or call the method implementation.
