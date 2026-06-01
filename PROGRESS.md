# Progress

## What Works

Implemented a scaled dynamic Gaussian splatting reproduction of Splannequin. The method uses time-dependent Gaussian primitive states, detects hidden states via camera-frustum visibility, detects defective states via near-zero reconstruction gradients, and anchors hidden states to randomly sampled past supervised states and defective states to randomly sampled future supervised states.

The benchmark generator, method code, baseline path, ablation variants, evaluator, scripts, briefing, reference metrics, and scientist-facing scoring documentation are in place.

Continuation audit on 2026-06-01 found that the workspace had no reproduced scores yet. The evaluator and method were adjusted before submission: the method renderer now matches the alpha-composited synthetic data renderer, and the evaluator now writes paper-named score groups in addition to scaled diagnostic metrics.

Continuation audit on 2026-06-01 verified that the existing implementation is a scaled implementation of the paper's computational steps rather than a surrogate. I added an extra-scene evaluation axis so the next job can test whether the core effect holds beyond the original single generated scene.

Continuation audit on 2026-06-01 after milestone reset confirmed the local fixed evaluator passes `python validate.py --compare`, and the last successful cluster job supports `core_claim`. Because the extra-scene evidence came from a job that failed only at validation output structure, I am submitting a clean CPU validation job before restoring `core_claim_plus`.

Clean validation job `f980876c-3fa` completed successfully on 2026-06-01. The workspace now has a clean cluster run for the fixed evaluator and restored `core_claim_plus`.

An integrity review later found that the previous scaled method used contribution-threshold defective detection and nearest-anchor sweeps instead of the paper's gradient-based defective state test and random reference timestamp procedure. I revised the method accordingly and am submitting a fresh CPU validation job. Until that job finishes, the current milestone is conservatively downgraded to `method_runs`.

## Results

The latest completed clean CPU job ran the earlier full scaled benchmark, regenerated outputs, and passed `python validate.py --compare`. These numbers are retained for traceability, but the revised implementation must be re-run before restoring a milestone above `method_runs`.

Scaled synthetic result from the latest completed job:

- Baseline dynamic Gaussian fixed-time PSNR: 17.3923
- Splannequin fixed-time PSNR: 17.7358
- Baseline LPIPS proxy: 0.0947
- Splannequin LPIPS proxy: 0.0838
- Baseline FVD proxy: 127.3078
- Splannequin FVD proxy: 102.4182
- Inference overhead: 0.0 for both

This supported the core claim for the previous implementation on the scaled benchmark.

The clean continuation CPU job also evaluated two extra generated scenes. Averaged across those extra scenes:

- Baseline PSNR: 17.1966; Splannequin PSNR: 18.1163
- Baseline SSIM proxy: 0.6316; Splannequin SSIM proxy: 0.6915
- Baseline LPIPS proxy: 0.0959; Splannequin LPIPS proxy: 0.0849
- Baseline FVD proxy: 118.2803; Splannequin FVD proxy: 95.1996
- Inference overhead: 0.0 for both

These extra-scene diagnostics supported `core_claim_plus` for the previous implementation.

## Remaining

Before restoring `core_claim` or `core_claim_plus`, the revised implementation needs a successful job that regenerates `checkpoints/`, rewrites `scoring/scores.json`, and passes `python validate.py --compare`.

## Issues

The exact paper benchmark videos and full dynamic Gaussian frameworks are not packaged here. The reproduction uses a controlled synthetic benchmark to keep the algorithm runnable within the available environment.

The login node lacks `numpy` and `torch`; dependency installation is therefore left to `environment/setup.sh` and the submitted container job.

The no-defective-loss ablation performed better than full Splannequin on the scaled diagnostic PSNR/ghosting metrics. This means the current reproduction does not claim the paper's secondary ablation result. Ablation numbers are kept in `scoring/diagnostics.json` only, not in `scoring/scores.json`.

The extra-scene job initially failed validation because an extra diagnostic experiment was written into `scoring/scores.json` without a paper-reference entry. The evaluator now keeps that robustness result in `scoring/diagnostics.json` only, and `python validate.py --compare` passes.

## Deviations from Paper

The paper trains full 2D/3D dynamic Gaussian splatting pipelines on 640x360 real Mannequin-Challenge videos and Blender scenes for 30,000 iterations. This reproduction uses a 1D differentiable Gaussian splatting scene with a forward camera path and known fixed-time ground truth. The deviation scales data dimensionality and runtime while preserving the paper's core computational steps: dynamic Gaussian primitive states, visibility and near-zero-gradient detection, hidden/past anchoring, defective/future anchoring, temporal-distance confidence weighting, and fixed-time rendering.

The paper reports SSIM, LPIPS, FVD, no-reference IQA, and COVER. The runnable evaluator uses PSNR plus artifact/stability proxy metrics suitable for the synthetic benchmark because heavyweight perceptual and video-quality models would require large external downloads and are unnecessary for the minimum core-claim check.
