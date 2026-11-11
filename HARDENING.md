# Hardening report (iteration 11)

## Summary

The audited gym could not reward science:
- **Copying the input solved it.** Its generated scene never moved, so the input frames were already the answer.
- **Its answers and seeds were visible.**
- **Self-reported numbers were trusted.** It used the method's own state counts.
- **Missing methods and scenes tied the reference.**
- **The scored reproduction ran in the wrong direction.** The reference lost to its own baseline on the hidden slice.

This iteration rebuilds the benchmark, the evaluators and the reference method together. All numbers below come from the final audit job (`6514bbd6-99d`). Every case was scored by the unmodified hidden evaluator (`eval/test/evaluate.py`, the program `scripts/evaluate_test.sh` runs), with the case's submission in the method slot. The driver is `redteam/hardening/audit.py`, with per-case records in `redteam/hardening/results/` and tables in `redteam/hardening/summary.{json,md}`.

**What changed**

- **Benchmark (`data/scenes.py`).**
  - Subjects now make smooth random unintended motions: Gaussian-process trajectories built from random Fourier features. Walls and backdrop stay static.
  - The camera moves forward with hand-held sway over 96 frames.
  - Rendering follows 3DGS conventions: front-to-back compositing, α ≤ 0.99, α < 1/255 skipped, termination at T < 1e-4. The old renderer composited back to front.
  - Two experiments: `synthetic_validation_paper` (near-static) and `synthetic_large_motion` (the paper's stated limitation).
- **Train/test split.**
  - Train: 6 public scenes per experiment, answers included.
  - Test: 16 hidden scenes per experiment, generated from private 62-bit seeds that live only in `eval/test/make_test_scenes.py`. Answers are in `eval/test/scenes/`; only inputs are in `data/test/`. Freeze instants are every 8th frame from a private per-scene offset (12 instants).
- **Submission.**
  - One frozen Gaussian state per timestamp, at most 64 primitives. The evaluator renders it with its own rasterizer from all camera poses, so it recomputes everything from raw outputs and never reads a reported number.
  - Exceeding the budget is the zero-inference-overhead gate. Any invalid, missing or out-of-range value also fails the method, which then gets the worst value of every metric in both experiments.
- **Slots.**
  - Every method directory on disk is scored.
  - The ranked slot `4dgaussians_splannequin` (`checkpoints/splannequin`) gets worst values if it is missing. Ambiguous slots fail.
  - `scripts/evaluate_test.sh` deletes the old score file, and the evaluator writes even when methods fail.
- **Metrics.**
  - PSNR is the per-view mean with a monotone 100 dB cap.
  - SSIM is windowed rather than global.
  - The LPIPS proxy is multi-scale.
  - FVD proxy and `edge_mae` are kept.
  - `foreground_mae` is dropped as redundant; `inference_overhead` became the gate.
  - New constraint metric `instant_selectivity`: the fraction of the true instant-specific change recovered, from −1 to 1. Static or black output scores 0 or below.
- **Reference method (`method/splannequin.py`), restored toward the paper.**
  - The per-timestamp parameter tables are replaced by a 4DGaussians-style model: canonical Gaussians plus a HexPlane deformation field with multires (1, 2) and time resolution 25, an MLP decoder, a coarse stage, and the K-planes TV / time-smoothness / L1-time regularizers. Training uses one random frame per iteration with L1 loss and 3DGS learning rates.
  - Splannequin runs at the paper's settings: 30k iterations, anchoring from 10k, L1 from 20k, λ = 10, τ = 5, two random view–timestamp pairs every 10 iterations. Hidden states use rasterizer visibility; defective states use a gradient norm ≤ 1e-9. Past anchors are used for hidden states and future anchors for defective ones.
  - The reference no longer receives true colors, opacities or scales as initialization. It gets only a noisy one-point-per-primitive cloud.
- **Phantom experiments removed.** `real_world_quality_improvement` (weight 0.35) and `ablation_regularizers` (0.20) were never written by the hidden test. Their paper numbers are kept unchanged in `scoring/reference.json` under `_paper_unscored_results`. The paper's Table 1 numbers are kept under `_paper_*` result keys, because they are not on this gym's scale.

## Do better results score higher now?

The rank delta is the stated contract: Σ weight·|coefficient|·direction·(case − reference)/sd. Here sd is each metric's spread over the 12 scientific cases of this audit, which is only illustrative because the real attempt population is unknown. "Paired" is the mean per-video PSNR difference to the reference over the same 192 freeze videos (16 scenes × 12 instants) per experiment, ± its standard error.

**Before** (REDTEAM, old gym):
- **Better variants were within seed noise.** Better initialization scored +0.186, 2× training +0.151 and longer/lower lr +0.037, against a seed range of 0.300.
- **Weaker variants scored too high.** 10-step training scored −0.060, and removing both anchors scored *+0.033*, above the reference.

**After:**

| case | paper PSNR | paper SSIM | paper sel. | large PSNR | large SSIM | large sel. | paired PSNR paper / large (dB) | rank Δ |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| reference (seed 0) | 24.381 | .8151 | .0332 | 21.044 | .7254 | .0511 | — | 0 |
| reference seed 1 | 24.615 | .8222 | .0512 | 20.896 | .7240 | .0501 | +0.23±.13 / −0.15±.09 | +1.265 |
| reference seed 2 | 24.514 | .8211 | .0383 | 21.002 | .7281 | .0434 | +0.13±.12 / −0.04±.08 | +0.980 |
| better: λ = 1 | 24.830 | .8230 | .0714 | 21.407 | .7314 | .0925 | **+0.45±.07 / +0.36±.05** | **+2.488** |
| better: time smoothness ×10 | 25.083 | .8328 | .0689 | 21.238 | .7307 | .0585 | **+0.70±.12 / +0.19±.05** | **+3.737** |
| better: λ = 1 + nearest-anchor snap | 24.822 | .8172 | .1023 | 21.453 | .7236 | .1134 | +0.44±.08 / +0.41±.08 | +2.282 |
| post-hoc nearest-anchor snap of reference | 24.434 | .8124 | .0657 | 21.146 | .7219 | .0735 | +0.05±.04 / +0.10±.05 | +0.294 |
| weaker: 4DGaussians baseline (no anchoring) | 24.051 | .7961 | .1179 | 21.056 | .7104 | .1366 | −0.33±.14 / +0.01±.10 | −2.633 |
| weaker: no hidden anchoring | 24.076 | .7969 | .1088 | 21.043 | .7102 | .1353 | −0.31±.15 / −0.00±.09 | −2.608 |
| baseline + nearest-anchor snap | 24.595 | .8098 | .1199 | 21.301 | .7130 | .1274 | +0.21±.13 / +0.26±.09 | +0.727 |
| static 3DGS-style reconstruction | 24.020 | .8412 | −.2945 | 18.793 | .7014 | −.2914 | −0.36±.21 / −2.25±.12 | −2.823 |
| trivial: render init point cloud | 16.264 | .4878 | −.3514 | 14.904 | .4161 | −.2980 | −8.12 / −6.14 | −64.687 |

The three reference seeds span 1.265 in rank and 0.23 / 0.15 dB in PSNR (paper / large). Both genuine better variants exceed that by a clear margin, in rank and in paired PSNR (≥ 5 standard errors on the paper experiment). The weaker variants fall below every reference seed.

The reproduction shows the paper's Table 1 direction on the near-static experiment. Splannequin beats its 4DGaussians baseline on PSNR (+0.33 dB paired), SSIM, the LPIPS proxy, the FVD proxy (23.8 vs 25.9) and edge error. Hidden anchoring carries the effect: no-hidden ≈ baseline.

The remaining test metrics (LPIPS / FVD / edge) are in `redteam/hardening/summary.json`. Train-slice values agree with the test values in scale and ordering. For example:

| | paper PSNR, train → test | large PSNR, train → test |
|---|---|---|
| reference | 23.46 → 24.38 | 21.14 → 21.04 |
| baseline | 23.32 → 24.05 | 20.82 → 21.06 |
| static | 23.16 → 24.02 | 19.33 → 18.79 |

An intermediate design (round 1: 48-frame videos, 8 test scenes per experiment, `redteam/hardening/round1/`) showed three problems:
- **A wrong constraint metric.** It used the error of the freeze-to-freeze differences. Static and black outputs scored *best* on it, because a noisy dynamic render's differences are worse than "no change".
- **Static outranked the reference** (+3.75).
- **Better variants (+0.9 to +1.45) stayed inside the seed range (2.47).**

The fixes were:
- `instant_selectivity`, which cannot be pushed below 0 by symmetric noise and is 0 for any instant-independent render;
- per-view PSNR;
- 96 frames, closer to the paper's ~240 frames per scene, with 4DGaussians' default time resolution;
- subtle-motion amplitude raised from 0.03–0.06 to 0.04–0.08;
- 16 test scenes per experiment.

## Major shortcuts: fix and score before → after

Before is the REDTEAM illustrative delta (its own normalization); after is this audit's illustrative rank delta. Every "after" result is a real hidden-evaluator run.

| shortcut | before | fix | after |
|---|---|---|---|
| copy visible ground truth | P = 99, +2.686 | Test answers exist only in `eval/test/scenes/`; `data/test/` holds inputs only. | not expressible (no test answers outside `eval/test/`) |
| copy input frames | P = 99, +2.686 (scene had no motion) | Subjects move. The output is one 3D state per instant, rendered from all poses, so frames cannot be submitted. | closest analogue (static fit): −2.823 |
| rebuild / brute-force generator seed | P = 99, +2.686 | Private 62-bit seeds; test scene IDs do not encode seeds. | enumerating 506 public seeds: PSNR 13.42 / 13.17, −88.574 |
| PSNR ε-discontinuity | 119.2 dB > exact copy | Per-view PSNR = min(100, −10·log10(max(MSE, 1e-10))), monotone. | exact answers 100.0; answers + 1e-4 jitter 72.6 |
| static reconstruction | +1.791 | Moving subjects, 12 private instants, `instant_selectivity` (weight 1.0), large-motion experiment. | −2.823 (selectivity −0.29; large-motion PSNR −2.25 dB) |
| false / negative / inflated state counts | +1e-7 / +0.459 / +1e-7 | Nothing self-reported is read; capacity is counted from the submitted arrays (≤ 64). | forged `inference_state_values` / metric fields: 0.000 (identical); 128 primitives: failed, worst values |
| replace fixed target | +3.018 | Targets are rendered from evaluator-held answers; `data/` and `checkpoints/` are never read as ground truth. | not expressible |
| black target + black outputs | +7.925 | Same as above. | black outputs: PSNR 7.88, −134.569 |
| null / infinite / empty outputs | NaN accepted / crash / crash | Strict validation; any failure gives the worst value of every metric in every experiment. | worst values (−2087) |
| omit the proposed method | tie (0) | Required slot `4dgaussians_splannequin` gets worst values when absent. | worst values (−2087) |
| failed proposed + boosted baseline slot | +5.240 | The proposed slot is always scored. Other slots are reported under their own names and are absent from the reference `scores_test.json`. | proposed slot (black) −134.569; baseline slot reported but unranked |
| extra method hidden | tie | Every method directory on disk is scored. | `junk_method` appears in results |
| drop extra scenes | tie | All 16 scenes per experiment are required. | one scene missing: worst values |
| stale perfect score file | file survived | The script removes it, the evaluator removes its output before scoring, and every method error is caught. | prewritten perfect file replaced by recomputed values |
| out-of-range opacity (new) | — | Range checks. | worst values |

## Shortcuts left, and why

- **Non-ranked slots.** Directories other than `splannequin` (for example `baseline/`) are scored under their own names. The reference `scores_test.json` contains only `4dgaussians_splannequin`, so they carry no rank. A ranker that ranked unknown slots would be outside this contract.
- **Per-scene search on test inputs.** The method must optimize on each test scene's frames; that is the task. A method can tune hyperparameters per scene against the photometric loss on visible frames. This is self-supervised reconstruction, and it cannot see the freeze-time objective.
- **Knowing the generator family.** The renderer is fixed and public. Subject motion is a smooth Gaussian-process sample, so a method can encode a temporal-smoothness prior. That is in scope (a "temporal prior"), and it cannot reach the bound, because unobserved motion is random. It was not tested beyond the time-smoothness ×10 variant.
- **Priors trained on extra generated scenes.** The public generator can produce unlimited scenes. Training a learned prior on them is forbidden by `scoring/CONSTRAINTS.md`, but it is not technically prevented. It is left to the shortcut grader.
- **Quality vs selectivity trade-off.** The 4DGaussians baseline has higher `instant_selectivity` (0.118) than the reference (0.033). A method may trade some quality for selectivity; this is an honest trade-off, priced by the coefficients.
- **Static wins some secondary metrics.** On the near-static experiment, static beats the reference on SSIM, the LPIPS proxy, the FVD proxy and edge error, consistent with the paper's Table 4 ("comparable to 3DGS"). It loses on PSNR, on selectivity in both experiments, and badly under large motion, so it ranks below the reference and below the dynamic baseline.

## Reference score before and after

- **Before** (old hidden slice: 1 scene, 12 views, global SSIM): PSNR 16.605, SSIM 0.660, LPIPS proxy 0.108, FVD proxy 130.4, edge 0.0395. The baseline scored higher on PSNR, SSIM and the LPIPS proxy.
- **After** (`scoring/scores_test.json`, reference seed 0):

| experiment | PSNR | SSIM | LPIPS proxy | FVD proxy | edge MAE | instant selectivity |
|---|---:|---:|---:|---:|---:|---:|
| `synthetic_validation_paper` | 24.381 | 0.8151 | 0.0449 | 23.79 | 0.0136 | 0.0332 |
| `synthetic_large_motion` | 21.044 | 0.7254 | 0.0678 | 30.46 | 0.0177 | 0.0511 |

The two tests are different benchmarks, so the numbers are not comparable. `scoring/scores_test.json` and `scoring/scores_train.json` were regenerated from the reference run alone with the no-argument scripts. `checkpoints/` holds only `splannequin/`.

## Headroom check

- **No scored metric sits at its bound.** Exact answers score PSNR 100, SSIM 1, error metrics 0 and selectivity 1. The reference is at 24.4 / 21.0 dB, SSIM 0.82 / 0.73, and selectivity 0.03 / 0.05.
- **A headroom experiment was added.** `synthetic_large_motion` (weight 0.3) comes from the paper's own limitation (Sec. 7). There the reference gains nothing over its baseline (+0.01 ± 0.10 dB), and both genuine better variants beat it (+0.36 and +0.19 dB paired).
- **Brute force fails.** Seed enumeration scores −88.6.
- **The simple baselines sit clearly below the reference:** init points −64.7, static −2.8, 4DGaussians −2.6. No simple variant reaches the bound.

## Weights

Experiment weights:
- `synthetic_validation_paper`: 0.45 → 0.7.
- `synthetic_large_motion`: new, 0.3.
- The phantom experiments (0.35 and 0.20) are removed.

Coefficients:
- `fvd` 0.4 → 0.5, because FVD is the paper's largest synthetic change (98.9 → 82.7).
- `instant_selectivity` is new, at 1.0. Instant selection is the capability the paper claims over static 3DGS (Sec. 6.4) and the only metric a static reconstruction fails.
- `foreground_mae` (redundant) and `inference_overhead` (now the gate) are removed.
- `psnr`, `ssim`, `lpips` and `edge_mae` are unchanged.

## Not fixed / limitations

- **Defective anchoring barely matters here.** Removing it changed PSNR by +0.10 ± 0.10 dB (paper) and −0.02 ± 0.01 dB (large) in round 1, so the paper's Table 3 claim that both anchors are necessary is not reproduced. Defective states are common: about 122k per 8-scene batch, against about 371k hidden states. But they rarely have a well-supervised future anchor in the sampled pair, so they make up only about 1% of anchored items (about 7.4k vs 772k hidden in the reference test runs). Ablations are not scored.
- **The reference trails the baseline on selectivity.** It suppresses part of the instant-specific change (selectivity 0.033 vs the baseline's 0.118), while staying far above static (−0.29).
- **The paper's λ = 10 is not optimal at this scale.** It is defined in scene units; λ = 1 is better. The reference keeps the paper's λ. The scale-downs are 2D scenes with 1D image lines, 64 primitives, no densification, and proxy LPIPS/FVD.
- **Real data are unmeasured.** Real-video results (CQA, COVER, IQA), the user study and runtime cannot be measured offline.
- **The ranking reference is the weakest of three seeds.** Reference seed 0 is the lowest of the three (seeds 1 and 2 rank +1.27 and +0.98), so an honest rerun of the reference method tends to score slightly above zero.
- **Compute.** Full reproduction (`scripts/run.sh`) trains 6 batches of up to 8 scenes for 30k iterations each, one single-threaded process per batch: about 30 minutes with 6 CPU cores. The hidden test needs about 2 minutes for one method (rendering the 384 target videos plus the submission).
