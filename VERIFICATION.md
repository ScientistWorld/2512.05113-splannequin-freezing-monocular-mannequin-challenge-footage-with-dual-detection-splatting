# Verification Report

## Initial Assessment

The workspace had previously claimed `core_claim_plus`, but an implementation-integrity review found genuine method-procedure issues: defective states were detected by rendered contribution rather than near-zero reconstruction gradient, and temporal anchors were selected by nearest-anchor sweeps rather than random reference timestamp sampling. Because the method code has now been revised, the old `scores.json` artifacts are stale and should not be used to restore `core_claim_plus` until a fresh job completes.

## Issues Found

- **Critical:** The previous scaled method did not match two key computational steps from the paper: gradient-based defective-state detection and random reference timestamp anchoring.
- **High:** The previous standard evaluator was too coupled to Splannequin ablation artifact names. It needed to accept any compatible prediction files and keep ablations optional.
- **High:** Stale scheduler artifacts under `jobs/` were still tracked by git.
- **Medium:** Current `scores.json`, `scores_train.json`, and `scores_test.json` were produced by the earlier implementation, not by the revised method.
- **Known limitation:** The benchmark is a scaled synthetic 1D environment rather than the paper's full real-video and Blender benchmark.

## Fixes Applied

- Revised `method/splannequin_1d.py` to use near-zero reconstruction gradients for defective detection, random reference timestamp sampling, staged L2-to-L1 anchoring, and time-varying position/color/opacity/scale parameters.
- Updated `eval/evaluate_freeze.py` and `scripts/evaluate.sh` so the standalone evaluator can score arbitrary compatible prediction files while keeping optional ablation diagnostics out of `scores.json`.
- Downgraded `MILESTONES.md` and `action.yaml` to `method_runs` pending a fresh validation run.
- Removed tracked `jobs/` scheduler artifacts and added `jobs/` to `.gitignore`.
- Re-ran lightweight local checks: Python syntax compilation, shell syntax checks, and `python validate.py --compare` on the stale pre-revision artifacts.

## Final Judgment

The workspace is not ready to signal final `done` in this turn because the revised implementation has not yet regenerated scored artifacts. The correct next step is the already configured CPU test submission, which will run `scripts/run.sh`, regenerate data/checkpoints/scores, run train/test evaluation, and validate the revised method.

- Milestone: method_runs
- Ready for gym use: partial
- Confidence: medium
- Key limitation: the revised method needs a fresh CPU validation job before any milestone above `method_runs` is justified.
