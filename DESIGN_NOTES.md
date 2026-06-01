# Train/Test Evaluation Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script invokes `eval/train/` and writes `scoring/scores_train.json`; the test script invokes `eval/test/` and writes `scoring/scores_test.json`.
- Checked that both split evaluators read method artifacts from `checkpoints/*/predictions.json` instead of regenerating the method inline.
- Grepped for train/test cross-imports and paper-agreement metric names. No cross-imports or paper-value deviation metrics were found.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stub exists.
- Ran `bash -n` on both train/test shell scripts.

## Changed

- Replaced the simple even/odd frame split with a blockwise camera-view split:
  - train: frame indices where `index mod 4` is `0` or `1`
  - test: frame indices where `index mod 4` is `2` or `3`
- Added two anti-shortcut metrics to both train and test slices:
  - `foreground_mae`, which focuses residual scoring on non-background target pixels so a background-biased renderer is less likely to look good.
  - `edge_mae`, which penalizes high-frequency residuals that expose blur and double-contour artifacts.
- Added top-level slice metadata to train/test score artifacts so generated outputs are not byte-identical in structure.
- Aligned train/test method names with the paper-facing reference names by mapping `checkpoints/baseline` to `4dgaussians` and `checkpoints/splannequin` to `4dgaussians_splannequin`.
- Added the required `description` field to every metric in `scoring/reference.json`, and declared the new `foreground_mae` and `edge_mae` metrics for `synthetic_validation_paper`.

## Runtime Concerns For Next Iteration

- The split still uses the same synthetic scene for train and test. The blockwise view split is stronger than parity, but it does not test scene-level generalization. The runner should watch for train/test gaps that are too small across all methods.
- `foreground_mae` uses a fixed target intensity threshold of `0.04`. This should be checked at runtime for saturation or excessive sparsity on the synthetic scene.
- `inference_overhead` depends on self-reported `inference_state_values` in prediction artifacts. That matches the existing artifact contract, but a future method could under-report it unless the method runner enforces honest metadata.
- The train/test scripts enumerate top-level `checkpoints/*/predictions.json`, so existing ablation directories such as `no_hidden` and `no_defective` will also appear in train/test score files. This is useful for sanity checks but may need filtering if the benchmark runner expects only headline methods.
