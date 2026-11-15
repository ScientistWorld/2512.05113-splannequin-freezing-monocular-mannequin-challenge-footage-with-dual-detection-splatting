# Hardening report (iteration 12)

## What this iteration had to fix

`REVIEW.md` (on the iteration-11 gym) listed two weaknesses:

1. **The reference depended on seed luck.** Reference seed 0 was the weakest of three seeds; another seed scored about +1.3 illustrative rank.
2. **The paper's defective-anchor ablation (Table 3) was not reproduced.**

The iteration-11 fixes for `REDTEAM.md` (private scenes, evaluator-rendered submissions, worst values on failure, and so on) are kept. They were re-verified on the final gym; see the shortcut table below.

All final numbers come from the unmodified hidden evaluator (`eval/test/evaluate.py`, the program `scripts/evaluate_test.sh` runs). Each case's submission sits in the ranked slot.
- **Driver:** `redteam/hardening_v2/audit.py` (train | cases | summary).
- **Per-case records:** `redteam/hardening_v2/results/`.
- **Tables:** `redteam/hardening_v2/summary.{json,md}`.
- **Jobs:** the final audit is job `9f8c7eb8-fde`, plus seed repeats in job `386da49f-4d0`.
- **Earlier steps** are kept for the record:
  - the design pilot `36f5182c-39b` (`redteam/hardening_v2/pilot.py`, `pilot_results.json`);
  - the first full audit `f590bd64-5a6` (`redteam/hardening_v2/run1_*`).

## Diagnosis

**Seed luck came from lost primitives.** Per-scene results swung by 2–6 dB between seeds. In the worst case (run-1 test scene 33), every seed-0 variant failed, including the unanchored baseline. So the failure came from the shared base training, not from anchoring.

The trained models had many "dead" primitives (opacity → 0, often with inflated scale). Seed 0 kept about 6 live primitives of the 18 that form the people near the start, and seed 1 about 9. The cause is the reproduction's fixed 64-primitive budget with *no* adaptive density control: a primitive that dies is never recovered. The paper states that it "follow[s] standard adaptive densification practices", so this was a departure from the paper.

**The defective ablation had nothing to act on.** In the iteration-11 scenes, walls and subjects were sparse, isolated primitives. In the ground truth only 0.05–1% of primitive-times were defective (in the frustum, zero contribution). Post-hoc snapping of the saved models' defective states to their future state changed PSNR by at most 0.04 dB (`tmp/analysis/defective_snap.py`).

The paper's 1e-9 gradient threshold made this worse. It is stated for the L1 loss of a 640×360 RGB frame (a mean over 691,200 values). On our 128×3 image line, per-pixel gradients are 1800× larger, so the literal threshold demands an exactly zero contribution.

**The anchoring gradient moved the anchor.** The anchoring loss pulled the ill-supervised state θ_k(t) toward a stop-gradient copy of θ_k(t_ref). The gradient also reached the canonical parameters, which θ_k(t_ref) shares. Moving them moves the anchor by exactly as much as the state, so it cannot reduce the discrepancy; it only displaces well-observed states.

This showed in run 1:
- Hidden anchoring *lowered* early-instant PSNR (reference − baseline −0.85 dB on the first third of freeze instants).
- No-defective was better on the middle and late instants.
- λ = 1 beat λ = 10 by +0.77 dB.

## What changed

### Benchmark (`data/scenes.py`, `eval/test/make_test_scenes.py`)

- **Crowded scenes.** Three people (the first three of six subject clusters: larger, nearly opaque bodies) stand across the walking line near the start. They hide the rest of the hallway until the camera has passed them, around frames 22–29. The primitive budget (64), motion model, renderer and both motion regimes are unchanged.
  - This is the paper's situation: Fig. 3, Sec. 5.1.2, "the camera has not yet reached it", and real Mannequin-Challenge footage with under 10% consistent visibility.
  - In the ground truth, 10–27% (literal threshold) or 31–43% (frame-normalised threshold) of first-third primitive-times are now defective with a later well-observed state, against 0–2% before.
- **Larger test.** 40 near-static scenes (was 16) and 16 large-motion scenes, from new private 62-bit seeds. Train is now 8 public scenes per experiment (was 6). The paper-regime test was enlarged because its per-scene seed noise is about twice that of the large-motion regime.

### Reference method (`method/splannequin.py`): restored toward the paper and its base framework

- **Adaptive density control** (3DGS/4DGaussians defaults), capped at the 64-primitive budget:
  - every 100 iterations from 500 to 15k, primitives with opacity < 0.005 are pruned;
  - their slots are refilled by splitting (scale/1.6) or cloning the primitives with the largest mean positional gradient;
  - opacity is reset to ≤ 0.01 every 3k iterations until 15k.
- **View sampling** as in 4DGaussians: random pops from a view stack without replacement, with a fresh stack for the coarse and fine stages. It was iid before.
- **Defective test at the paper's frame size.** The paper's 1e-9 threshold is applied to the gradient rescaled to a 640×360×3-value loss (`--grad-loss-values 691200`, the default; `0` gives the literal threshold).
  - Pilot (8 scenes): with the literal threshold, no-defective − full method was +0.04 ± 0.06 dB (no effect). With the frame-normalised one it was −0.15 ± 0.11 dB, and −0.53 dB on early instants.
- **Fixed anchor.** The anchoring gradient reaches the deformation of the ill-supervised state but not the canonical parameters it shares with the anchor. `--anchor-canonical-grad` restores the iteration-11 behaviour.
  - The paper does not say where the gradient stops. Its "stable reference state" is what this enforces.
  - Measured effect: the old behaviour scores −0.75 ± 0.07 dB (paper regime) and −0.45 ± 0.09 dB (large motion), rank −4.2.

The paper's schedule and loss are unchanged: 30k iterations, anchoring from 10k, L2 → L1 at 20k, λ = 10, τ = 5, two view–timestamp pairs every 10 iterations, hidden → past and defective → future anchors.

### Evaluators (`eval/train/evaluate.py`, `eval/test/evaluate.py`; still no shared code)

- **New gate.** A frozen state identical at every timestamp of a scene cannot select an instant. That is a failed constraint, and the whole run gets worst values.
- **Selectivity docstring corrected.** Any instant-independent render scores *at most* 0 on `instant_selectivity` (by convexity); it was described as exactly 0.

### Scoring

- `scoring/reference.json` holds the final reference and baseline test values. The paper's numbers are unchanged under `_paper_*` / `_paper_unscored_results`.
- Experiment weights (0.7 / 0.3) and metric coefficients are unchanged from iteration 11. A coefficient of 1.5 for `instant_selectivity` was tried after the pilot (where static came close) and reverted: with the fixed anchor, static ranks far below the reference at 1.0 (−3.3), and the exact-static gate handles the pure shortcut.

## Do better results score higher now?

The rank delta is the stated contract: Σ weight·|coefficient|·direction·(case − reference)/sd. Here sd is each metric's spread over the 13 scientific cases of this audit, so it is illustrative only. "Paired" is the mean per-scene PSNR difference to the reference over the same 40 / 16 test scenes, ± its standard error over scenes.

**Before (iteration-11 audit):**
- Reference seed 0 was the weakest seed. Seeds 1 and 2 scored +1.27 and +0.98, and the per-scene seed spread was 0.64 dB.
- Better variants: λ = 1 +2.49, time smoothness ×10 +3.74.
- Weaker variants: baseline −2.63, no-hidden −2.61, static −2.82.
- No-defective was not distinguishable (+0.10 ± 0.10 dB).

**After (final gym):**

| case | paper PSNR | paper SSIM | paper FVD | paper sel. | large PSNR | large SSIM | large sel. | paired PSNR vs reference, paper / large (dB) | rank Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| reference (seed 0) | 24.348 | 0.8126 | 24.88 | −0.0479 | 20.571 | 0.7181 | +0.0052 | — | 0 |
| reference seed 1 | 24.284 | 0.8104 | 25.33 | −0.0450 | 20.424 | 0.7183 | +0.0003 | −0.06 ± 0.19 / −0.15 ± 0.15 | −0.506 |
| reference seed 2 | 24.464 | 0.8170 | 24.74 | −0.0440 | 20.719 | 0.7276 | +0.0097 | +0.12 ± 0.16 / +0.15 ± 0.13 | +0.869 |
| **better: time smoothness ×10** (seed 0) | 24.626 | 0.8190 | 24.91 | −0.0172 | 20.723 | 0.7268 | +0.0213 | +0.28 ± 0.09 / +0.15 ± 0.13 | **+1.541** |
| better: time smoothness ×10, seed 1 | 24.862 | 0.8241 | 24.33 | −0.0155 | 20.812 | 0.7272 | +0.0296 | +0.51 ± 0.15 / +0.24 ± 0.14 | **+2.547** |
| better: time smoothness ×10, seed 2 | 24.916 | 0.8260 | 24.35 | −0.0205 | 20.956 | 0.7362 | +0.0197 | +0.57 ± 0.15 / +0.38 ± 0.18 | **+2.912** |
| post-hoc nearest-anchor snap of the reference | 24.412 | 0.8119 | 24.71 | −0.0262 | 20.619 | 0.7169 | +0.0236 | +0.06 ± 0.05 / +0.05 ± 0.07 | +0.493 |
| λ = 3 | 24.325 | 0.8097 | 25.08 | −0.0356 | 20.541 | 0.7146 | +0.0149 | −0.02 ± 0.05 / −0.03 ± 0.04 | −0.196 |
| λ = 1 | 24.251 | 0.8072 | 25.33 | −0.0310 | 20.594 | 0.7158 | +0.0229 | −0.10 ± 0.06 / +0.02 ± 0.10 | −0.374 |
| λ = 30 | 24.375 | 0.8126 | 24.69 | −0.0600 | 20.246 | 0.7175 | −0.0162 | +0.03 ± 0.07 / −0.33 ± 0.19 | −0.386 |
| weaker: no defective anchoring (seed 0) | 24.310 | 0.8090 | 25.15 | −0.0263 | 20.397 | 0.7128 | +0.0128 | −0.04 ± 0.06 / −0.17 ± 0.10 | −0.367 |
| weaker: old anchoring (gradient into canonical) | 23.597 | 0.7923 | 27.04 | −0.0833 | 20.122 | 0.7030 | −0.0275 | −0.75 ± 0.07 / −0.45 ± 0.09 | −4.176 |
| weaker: no hidden anchoring | 23.494 | 0.7799 | 28.26 | −0.0039 | 19.719 | 0.6744 | +0.0226 | −0.85 ± 0.12 / −0.85 ± 0.15 | −5.706 |
| weaker: 4DGaussians baseline (no anchoring) | 23.337 | 0.7753 | 28.47 | +0.0042 | 19.703 | 0.6718 | +0.0390 | −1.01 ± 0.12 / −0.87 ± 0.18 | −6.246 |
| static 3DGS-style + 1e-6 nudge (evades the gate) | 23.974 | 0.8412 | 23.20 | −0.2747 | 18.999 | 0.7237 | −0.2866 | −0.37 ± 0.21 / −1.57 ± 0.30 | −3.306 |
| trivial: render the init point cloud (+ nudge) | 15.768 | 0.4763 | 40.24 | −0.2819 | 14.134 | 0.3980 | −0.2714 | −8.58 / −6.44 | −55.403 |

**Seed spread.**
- The three reference seeds span 1.37 in rank (−0.51 … +0.87), and seed 0 sits in the middle.
- In raw metrics the spread is 0.18 dB / 0.30 dB PSNR and 0.0066 / 0.0095 SSIM (paper / large motion).

**The better variant beats the reference beyond seed noise.** Each of its three seeds outranks every reference seed (+1.54, +2.55, +2.91 against at most +0.87). Seed-matched over three seeds (variant seed k vs reference seed k, 120 / 48 paired scenes):
- PSNR: +0.44 ± 0.06 dB / +0.26 ± 0.07 dB;
- SSIM: +0.0097 ± 0.0017 / +0.0087 ± 0.0025;
- LPIPS proxy: −0.0022 ± 0.0005 / −0.0028 ± 0.0007;
- edge error: −0.0004 ± 0.0001 / −0.0003 ± 0.0001;
- selectivity: +0.028 ± 0.003 / +0.019 ± 0.005;
- illustrative rank of the three-seed means: +2.21.

**Weaker variants fall below every reference seed.** The baseline, no-hidden and old anchoring score −6.2, −5.7 and −4.2.

**The paper's main claim reproduces strongly and the right way round** (Table 1: 4DGaussians + Splannequin vs 4DGaussians):
- **Near-static experiment.** PSNR +1.01 ± 0.12 dB, better on 93% of scenes. SSIM +0.037, LPIPS proxy −0.008 and FVD proxy −3.6 (24.9 vs 28.5; paper 82.7 vs 98.9). Edge error is lower too.
- **Large-motion experiment.** +0.87 dB, with the same direction on every quality metric.
- **Train slice agrees.** Reference vs baseline PSNR is 25.02 vs 23.90 (near-static) and 21.58 vs 21.80 (large motion, 8 scenes). On the large-motion train slice the reference is better on SSIM, FVD and edge error but not on PSNR or LPIPS.

**Ablations (Table 3):**
- **No hidden anchoring:** −0.85 dB in both experiments. Reproduced.
- **No defective anchoring:** the paper's direction on quality, but small. Seed-matched over three seeds:
  - PSNR −0.02 ± 0.03 / −0.06 ± 0.04 dB;
  - early-instant PSNR, where future anchors act, −0.15 ± 0.08 / −0.17 ± 0.08 dB;
  - SSIM −0.0019 ± 0.0010 / −0.0042 ± 0.0010;
  - LPIPS proxy +0.0004 ± 0.0003 / +0.0009 ± 0.0004.
- No-defective has slightly better selectivity (+0.016 ± 0.003), so its three-seed illustrative rank is a tie (−0.06).

**Intermediate run 1** (crowded scenes and frame-normalised threshold, before density control and the fixed anchor):
- reference − baseline was +0.03 ± 0.17 dB;
- no-defective scored +0.15 ± 0.10 dB *above* the reference;
- reference seed 0 was again the weakest seed (24.42 vs 24.49 / 24.73).

The diagnosis above came from that run.

## Major shortcuts: fix and score before → after

"Before" is the iteration-11 audit (illustrative delta, its own normalization). "After" is this audit; every value is a real hidden-evaluator run on the final gym.

| shortcut | before | fix | after |
|---|---|---|---|
| exact static reconstruction (same state at every timestamp) | −2.823 (ranked below the reference only through `instant_selectivity`) | **new gate:** a state identical at every timestamp of a scene fails the run | failed, worst values (−1616) |
| static + 1e-6 nudge (evades the gate) | — | coefficients and the large-motion experiment | −3.306 (paper SSIM/LPIPS/FVD better than the reference, consistent with the paper's Table 4; PSNR −0.37 / −1.57 dB, selectivity −0.27 / −0.29) |
| enumerate public generator seeds | −88.574 | private 62-bit seeds (new for the new scenes) | −66.434 (best public seed's frame MSE ≈ 0.04, no match) |
| null / infinite / empty predictions, over budget, out-of-range opacity, missing scene, omitted proposed slot | worst values | unchanged | worst values (−1616) |
| black output / failed proposed + boosted baseline slot | −134.6 | unchanged | −107.6; the baseline slot is reported but unranked |
| forged metric fields, stale perfect score file, extra junk method | 0.000 | unchanged | 0.000 (identical to the reference) |
| PSNR ε-discontinuity | exact answers 100, +1e-4 jitter 72.6 | unchanged | 100.0 / 73.4 |

Test answers exist only in `eval/test/scenes/` (and the seeds only in `eval/test/make_test_scenes.py`); `data/test/` holds inputs only.

## Shortcuts left, and why

- **Static + a negligible per-timestamp change.** It evades the exact gate and scores −3.3, below the reference and every honest dynamic variant. A tolerance-based gate could fail honest near-static methods, so this is left to the shortcut grader. `scoring/DIRECTION.md` puts static reconstruction out of bounds.
- **Seed choice.** Rerunning the reference with another seed moves it within about −0.5 … +0.9 illustrative rank. A future agent cannot pick seeds on test scores, which are hidden. Picking on train scores is train tuning, and train and test agree in direction.
- **Unchanged from iteration 11:**
  - non-ranked slots (scored under their own names, absent from the reference `scores_test.json`);
  - per-scene self-supervised fitting on the test inputs (that is the task);
  - a temporal-smoothness prior from knowing the generator family (in scope, and it cannot reach the bound);
  - priors trained on extra generated scenes (forbidden in `scoring/CONSTRAINTS.md`, not technically prevented);
  - the honest quality-vs-selectivity trade-off: the baseline has higher selectivity than the reference.

## Reference score before and after

| | paper PSNR | SSIM | LPIPS proxy | FVD proxy | edge | selectivity | large PSNR | SSIM | LPIPS | FVD | edge | selectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before (iteration 11, 16 + 16 old scenes) | 24.381 | 0.8151 | 0.0449 | 23.79 | 0.0136 | 0.0332 | 21.044 | 0.7254 | 0.0678 | 30.46 | 0.0177 | 0.0511 |
| after (`scoring/scores_test.json`, 40 + 16 new scenes) | 24.348 | 0.8126 | 0.0468 | 24.88 | 0.0121 | −0.0479 | 20.571 | 0.7181 | 0.0763 | 33.19 | 0.0166 | 0.0052 |

The scenes differ, so the rows are not comparable as numbers. `scoring/scores_test.json` and `scoring/scores_train.json` (and `scores.json`) were regenerated from the reference alone with the no-argument scripts. `checkpoints/` holds only `splannequin/`.

## Headroom check

- **No scored metric sits at its bound.** Exact answers score PSNR 100, SSIM 1, errors 0 and selectivity 1. The reference is at 24.3 / 20.6 dB, SSIM 0.81 / 0.72, and selectivity −0.05 / 0.01.
- **The headroom experiment `synthetic_large_motion` is kept** (weight 0.3, the paper's stated limitation).
- **A genuine better variant exists and is seed-robust:** +2.2 illustrative rank (three-seed mean), above every reference seed. No simple variant reaches the bound.
- **Brute force fails** (−66). The simple baselines are clearly below the reference: init points −55, static −3.3, 4DGaussians −6.2.

## Not fixed / limitations

- **The defective ablation is only partly reproduced.** It goes the paper's way on quality, mostly on the early instants where future anchors act, at 2–4 standard errors on SSIM. But its PSNR effect is small and its selectivity is slightly worse, so its rank ties the full method. The paper's finding that removing defective anchoring hurts *more* than removing hidden anchoring (Table 3, real-video NR metrics) is not reproduced: here hidden ≫ defective.
- **Seed noise is reduced, not removed.** The seed range (1.37 rank, 0.18 dB) is well below the paper's effect (6.2) and below the better variant's gain, but a single run of a marginally better method (≲ +0.2 dB) cannot be told from seed luck.
- **Reproduction across machines.** Training is single-threaded and seeded, but the audit ran on several (pre-empted) nodes. A rerun of `scripts/run.sh` on different CPU hardware can diverge at the level of seed noise.
- **λ.** The paper's λ = 10 is kept. With the fixed anchor, λ = 1, 3 and 30 are all slightly worse than or equal to it, so the earlier "λ = 1 is better" finding was an artefact of the moving anchor.
- **Real data are unmeasured.** Real-video results (CQA, COVER, IQA), the user study and runtime cannot be measured offline. Their paper numbers are kept unscored in `scoring/reference.json`.
- **Scale-downs:** 2D scenes with 1D image lines, 64 primitives (densification capped), and proxy LPIPS/FVD.
- **Compute.** `scripts/run.sh` trains the reference in 9 concurrent single-threaded batches of up to 8 scenes (8 + 8 train, 40 + 16 test), 30k iterations each: about 15–25 minutes with 8 cores. The hidden test takes about 1 minute per method.
