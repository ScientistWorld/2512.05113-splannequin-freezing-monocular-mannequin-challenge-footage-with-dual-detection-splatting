# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-01 10:32] - core_claim
- The implemented dynamic Gaussian splatting reproduction runs end-to-end on the scaled forward-camera benchmark.
- Splannequin's hidden/defective dual detection, past/future anchoring, temporal confidence weighting, and fixed-time rendering improve the synthetic fixed-time result over the baseline dynamic Gaussian reconstruction.
- `python validate.py --compare` passes, and `scoring/scores.json` records the reproduced metrics.

## Stop Justification
- Completed at milestone `core_claim`.
