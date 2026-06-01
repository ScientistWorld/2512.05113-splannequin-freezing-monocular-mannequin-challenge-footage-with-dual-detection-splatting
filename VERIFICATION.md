# Verification Report

## Initial Assessment

The workspace started at `core_claim_plus`. The claimed milestone was broadly justified: the scaled method had completed a clean CPU job, `scores.json` showed the proposed fixed-time reconstruction improving over the dynamic baseline on the primary synthetic benchmark, and `python validate.py --compare` passed.

## Issues Found

- **Critical:** Strict method-agnostic documentation leaked representation and algorithm details in `briefing/evaluation.md` and `scoring/EXPERIMENTS.md`.
- **High:** `scores.json` omitted `foreground_mae` and `edge_mae` even though they were weighted in `reference.json`.
- **High:** `scripts/run.sh` assumed ignored checkpoint artifacts were already present.
- **High:** Tracked scheduler logs and trajectory traces were stale runtime artifacts.
- **Medium:** Train/test evaluators defaulted to the method keys already present in `scores.json`, which could ignore future checkpoint subdirectories.
- **Medium:** The ablation degradation metrics in `reference.json` had an inconsistent direction for negative degradation values.
- **Known limitation:** The held-out test split is public and derives from the same synthetic file, so it is useful as a consistency check but not as a secure hidden benchmark.

## Fixes Applied

- Rewrote method-agnostic and method-safe documentation to describe the task and scoring without leaking the paper's internal mechanism.
- Added `foreground_mae` and `edge_mae` to the standalone evaluator output and to `scoring/scores.json`.
- Updated `scripts/run.sh` to call `scripts/reproduce.sh` before train/test scoring.
- Removed tracked `jobs/` and `trajectory/` artifacts from the git index; they remain ignored runtime files.
- Updated train/test discovery so future methods with compatible `checkpoints/*/predictions.json` outputs are scored instead of silently ignored.
- Corrected ablation metric direction metadata for negative degradation values.
- Re-ran `python validate.py --compare`; all checks passed.

## Final Judgment

This workspace is ready for gym use at the claimed milestone. From-scratch agents can use the method-agnostic problem, data, eval, and scoring docs without seeing the paper's internal algorithm. Improve-mode agents can inspect the included scaled implementation and compare against the stored baseline scores.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: medium
- Key limitation: the reproduction uses a scaled public synthetic benchmark rather than the paper's full real-video benchmark, and the train/test split is not a private anti-cheat split.
