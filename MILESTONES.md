# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-01 10:32] - core_claim
- The implemented dynamic Gaussian splatting reproduction runs end-to-end on the scaled forward-camera benchmark.
- Splannequin's hidden/defective dual detection, past/future anchoring, temporal confidence weighting, and fixed-time rendering improve the synthetic fixed-time result over the baseline dynamic Gaussian reconstruction.
- `python validate.py --compare` passes, and `scoring/scores.json` records the reproduced metrics.

### [2026-06-01 10:32] - core_claim
- Continuation audit rechecked method/eval/data/scripts and restored `core_claim`: the implementation contains the paper's actual dual-detection temporal anchoring steps, and evaluation reads saved method outputs independently.
- Added a pending extra-scene benchmark to test robustness beyond the minimum core-claim scene.

## Stop Justification
- Not stopping yet. A CPU job is pending to test whether the reproduction can advance from `core_claim` to `core_claim_plus`.
