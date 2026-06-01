# Reproduction Milestones

**Current: core_claim_plus**

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
