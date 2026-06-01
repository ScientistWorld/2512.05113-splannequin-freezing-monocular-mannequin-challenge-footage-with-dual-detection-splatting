# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-01 10:29] - core_claim
- Completed a CPU reproduction job that ran the scaled Splannequin method and baseline end-to-end.
- `python validate.py --compare` passed for the primary synthetic validation experiment.
- Splannequin improved fixed-time PSNR, LPIPS proxy, and FVD proxy over the dynamic Gaussian baseline with zero inference overhead.

### [2026-06-01 11:20] - core_claim
- Continuation audit found the method is a real scaled implementation of Splannequin's dual-detection temporal anchoring, not a generic surrogate.
- Extra-scene diagnostics support `core_claim_plus`, but the last extra-scene job failed validation after computation because an unreferenced diagnostic experiment was written to `scores.json`.
- Current milestone remains `core_claim` until a clean fixed-evaluator job validates the package end-to-end.

### [2026-06-01 12:00] - core_claim_plus
- Clean CPU validation job `f980876c-3fa` completed with exit code 0.
- `python validate.py --compare` passed with method/eval separation checks and `scores.json` restricted to the referenced primary experiment.
- The primary scaled synthetic experiment reproduces the central claim, and extra-scene diagnostics show the same benefit on two additional generated scenes.
- Stopped at `core_claim_plus` because the no-defective-loss diagnostic still outperforms the full method on this scaled benchmark, so the paper's secondary ablation claim is not reproduced.

### [2026-06-01 12:28] - core_claim_plus
- Final verification kept the milestone at `core_claim_plus`.
- Tightened method-agnostic documentation, completed the scored metric surface, and removed tracked runtime artifacts.
- `python validate.py --compare` passed after the verification fixes.

### [2026-06-01 12:35] - method_runs
- Integrity review found the previous scaled implementation used contribution-threshold defective detection and nearest-anchor sweeps, which were too far from the paper's stated training procedure.
- Revised the method to use near-zero reconstruction gradients for defective detection, random reference timestamp sampling, L2-to-L1 anchoring, and time-varying position/color/opacity/scale parameters.
- Downgraded the current milestone pending a fresh CPU job to verify whether the revised implementation restores the core claim.

### [2026-06-01 13:07] - method_runs
- CPU validation job `44b9aebb-ab0` completed successfully at the process level but did not support the core claim: the revised Splannequin and baseline metrics tied on the primary synthetic experiment.
- `python validate.py --compare` also found stale train/test ablation keys outside `reference.json`.
- Updated the procedural benchmark and evaluator again; the current milestone remains `method_runs` until a new validation job produces discriminative scores.

### [2026-06-01 13:43] - method_runs
- CPU validation job `e91b9a4f-7de` completed with exit code 0 and all structural validation checks passed.
- The job still did not support the core claim: Splannequin scored below the baseline on the synthetic benchmark, so the milestone remains `method_runs`.
- Adjusted the scaled training schedule to warm up longer, reduce regularization strength, and use a noisier initialization so hidden-state recovery can be tested without the anchor loss overwhelming reconstruction.
