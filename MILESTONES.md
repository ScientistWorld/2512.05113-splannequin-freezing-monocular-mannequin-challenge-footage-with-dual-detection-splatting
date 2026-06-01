# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-01 10:32] - core_claim
- The implemented dynamic Gaussian splatting reproduction runs end-to-end on the scaled forward-camera benchmark.
- Splannequin's hidden/defective dual detection, past/future anchoring, temporal confidence weighting, and fixed-time rendering improve the synthetic fixed-time result over the baseline dynamic Gaussian reconstruction.
- `python validate.py --compare` passes, and `scoring/scores.json` records the reproduced metrics.

### [2026-06-01 10:32] - core_claim
- Continuation audit rechecked method/eval/data/scripts and restored `core_claim`: the implementation contains the paper's actual dual-detection temporal anchoring steps, and evaluation reads saved method outputs independently.
- Added a pending extra-scene benchmark to test robustness beyond the minimum core-claim scene.

### [2026-06-01 10:58] - core_claim_plus
- The CPU job regenerated the core benchmark and trained/evaluated two additional synthetic forward-camera scenes.
- Extra-scene average improved from baseline to Splannequin on PSNR, SSIM, LPIPS proxy, and FVD proxy while preserving zero inference overhead.
- Fixed score packaging so `scoring/scores.json` contains only experiments present in `scoring/reference.json`; extra-scene robustness numbers are kept in `scoring/diagnostics.json`.

## Stop Justification
- Completed at milestone `core_claim_plus`. Higher milestones would require full dynamic Gaussian integrations or real-video/no-reference quality evaluation beyond the scaled benchmark.
