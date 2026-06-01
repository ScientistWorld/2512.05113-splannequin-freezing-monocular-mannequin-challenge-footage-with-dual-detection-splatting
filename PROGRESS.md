# Progress

## What Works

Implemented a scaled dynamic Gaussian splatting reproduction of Splannequin. The method uses time-dependent Gaussian primitive states, detects hidden states via camera-frustum visibility, detects defective states via negligible rendered contribution, and anchors hidden states to past supervised states and defective states to future supervised states.

The benchmark generator, method code, baseline path, ablation variants, evaluator, scripts, briefing, reference metrics, and scientist-facing scoring documentation are in place.

Continuation audit on 2026-06-01 found that the workspace had no reproduced scores yet. The evaluator and method were adjusted before submission: the method renderer now matches the alpha-composited synthetic data renderer, and the evaluator now writes paper-named score groups in addition to scaled diagnostic metrics.

## Results

The first CPU job ran the full scaled benchmark and wrote real outputs. It failed only because `scores.json` included an extra diagnostic experiment not listed in `reference.json`; the score contract has been fixed for the next submission.

Scaled synthetic result from the completed job:

- Baseline dynamic Gaussian fixed-time PSNR: 17.3923
- Splannequin fixed-time PSNR: 17.7358
- Baseline LPIPS proxy: 0.0947
- Splannequin LPIPS proxy: 0.0838
- Baseline FVD proxy: 127.3078
- Splannequin FVD proxy: 102.4182
- Inference overhead: 0.0 for both

This supports the core claim on the scaled benchmark, but I am resubmitting after the score-format fix before marking `core_claim`.

## Remaining

Resubmit the packaged reproduction job with the corrected score contract. If validation passes and the same improvement is reproduced, update to `core_claim`.

## Issues

The exact paper benchmark videos and full dynamic Gaussian frameworks are not packaged here. The reproduction uses a controlled synthetic benchmark to keep the algorithm runnable within the available environment.

The login node lacks `numpy` and `torch`; dependency installation is therefore left to `environment/setup.sh` and the submitted container job.

## Deviations from Paper

The paper trains full 2D/3D dynamic Gaussian splatting pipelines on 640x360 real Mannequin-Challenge videos and Blender scenes for 30,000 iterations. This reproduction uses a 1D differentiable Gaussian splatting scene with a forward camera path and known fixed-time ground truth. The deviation scales data dimensionality and runtime while preserving the paper's core computational steps: dynamic Gaussian primitive states, visibility/contribution detection, hidden/past anchoring, defective/future anchoring, temporal-distance confidence weighting, and fixed-time rendering.

The paper reports SSIM, LPIPS, FVD, no-reference IQA, and COVER. The runnable evaluator uses PSNR plus artifact/stability proxy metrics suitable for the synthetic benchmark because heavyweight perceptual and video-quality models would require large external downloads and are unnecessary for the minimum core-claim check.
