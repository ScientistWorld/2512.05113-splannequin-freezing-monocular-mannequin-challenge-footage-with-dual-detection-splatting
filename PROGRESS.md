# Progress

## What Works

Implemented a scaled dynamic Gaussian splatting reproduction of Splannequin. The method uses time-dependent Gaussian primitive states, detects hidden states via camera-frustum visibility, detects defective states via negligible rendered contribution, and anchors hidden states to past supervised states and defective states to future supervised states.

The benchmark generator, method code, baseline path, ablation variants, evaluator, scripts, briefing, reference metrics, and scientist-facing scoring documentation are in place.

Continuation audit on 2026-06-01 found that the workspace had no reproduced scores yet. The evaluator and method were adjusted before submission: the method renderer now matches the alpha-composited synthetic data renderer, and the evaluator now writes paper-named score groups in addition to scaled diagnostic metrics.

## Results

No reproduced scores yet. The next submitted CPU job will install dependencies in the container overlay, run baseline/Splannequin/ablations, write `scoring/scores.json`, and run `python validate.py --compare`.

Paper reference numbers have been recorded in `scoring/reference.json`.

## Remaining

Run the packaged reproduction job and inspect whether Splannequin improves fixed-time PSNR and artifact metrics over the unregularized dynamic Gaussian baseline. If the effect is weak, tune only scale-preserving hyperparameters and resubmit.

## Issues

The exact paper benchmark videos and full dynamic Gaussian frameworks are not packaged here. The reproduction uses a controlled synthetic benchmark to keep the algorithm runnable within the available environment.

The login node lacks `numpy` and `torch`; dependency installation is therefore left to `environment/setup.sh` and the submitted container job.

## Deviations from Paper

The paper trains full 2D/3D dynamic Gaussian splatting pipelines on 640x360 real Mannequin-Challenge videos and Blender scenes for 30,000 iterations. This reproduction uses a 1D differentiable Gaussian splatting scene with a forward camera path and known fixed-time ground truth. The deviation scales data dimensionality and runtime while preserving the paper's core computational steps: dynamic Gaussian primitive states, visibility/contribution detection, hidden/past anchoring, defective/future anchoring, temporal-distance confidence weighting, and fixed-time rendering.

The paper reports SSIM, LPIPS, FVD, no-reference IQA, and COVER. The runnable evaluator uses PSNR plus artifact/stability proxy metrics suitable for the synthetic benchmark because heavyweight perceptual and video-quality models would require large external downloads and are unnecessary for the minimum core-claim check.
