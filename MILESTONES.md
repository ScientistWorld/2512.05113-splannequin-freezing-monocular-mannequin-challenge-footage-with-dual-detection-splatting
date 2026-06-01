# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-01 10:29] - core_claim
- Completed a CPU reproduction job that ran the scaled Splannequin method and baseline end-to-end.
- `python validate.py --compare` passed for the primary synthetic validation experiment.
- Splannequin improved fixed-time PSNR, LPIPS proxy, and FVD proxy over the dynamic Gaussian baseline with zero inference overhead.

### [2026-06-01 11:20] - core_claim
- Continuation audit found the method is a real scaled implementation of Splannequin's dual-detection temporal anchoring, not a generic surrogate.
- Extra-scene diagnostics support `core_claim_plus`, but the last extra-scene job failed validation after computation because an unreferenced diagnostic experiment was written to `scores.json`.
- Current milestone remains `core_claim` until a clean fixed-evaluator job validates the package end-to-end.
