# Train/Test Evaluation Design Notes (current, iteration 12)

- Visible train slice: `scripts/evaluate_train.sh` → `eval/train/evaluate.py`. It scores 8 public scenes per experiment (`data/train`, answers included) and writes `scoring/scores_train.json` and `scoring/diagnostics_train.json`.
- Hidden test slice: `scripts/evaluate_test.sh` → `eval/test/evaluate.py`. It scores 40 (`synthetic_validation_paper`) and 16 (`synthetic_large_motion`) private scenes. Answers are in `eval/test/scenes`; inputs are in `data/test`; seeds are only in `eval/test/make_test_scenes.py`. Freeze instants use a private offset. Output goes to `scoring/scores_test.json`.
- Both slices use the same interface. They score every `checkpoints/<method>/predictions/` directory by rendering the submitted frozen Gaussian states with their own rasterizer. Neither slice imports `method/` or shares code with the other.
- Validity gate: ≤ 64 primitives, finite and in-range values, every scene present, and a frozen state that is not identical at every timestamp (instant selection). A failure gives worst values in every metric of every experiment. The required slot `4dgaussians_splannequin` gets worst values when absent. Stale score files are removed before scoring.
- Metrics: per-view PSNR (monotone 100 dB cap), windowed SSIM, a multi-scale MAE (LPIPS proxy), the cross-view temporal-derivative error (FVD proxy), edge MAE, and instant selectivity (the constraint).
- The audit and its evidence are in `HARDENING.md` and `redteam/hardening_v2/` (iteration 11: `redteam/hardening/`).
