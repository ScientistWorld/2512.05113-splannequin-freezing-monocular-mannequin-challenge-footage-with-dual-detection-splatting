# Evaluation Targets

## Primary: synthetic_freeze_time

Tests whether fixed-time renderings are closer to a static target than a dynamic reconstruction rendered naively at one timestamp.

- **Metric**: `freeze_psnr` (higher is better)
- **Metric**: `ghosting_error` (lower is better)
- **Metric**: `temporal_error` (lower is better)
- **How to run**: `bash scripts/reproduce.sh` or `bash scripts/evaluate.sh` after writing compatible prediction files.

## Constraint: zero_inference_overhead

Tests whether quality improvements are achieved without increasing the state required at rendering time.

- **Metric**: `inference_state_values` (lower is better; proposed and baseline should match)
- **How to run**: included in `bash scripts/evaluate.sh`.

## Ablation: component_removal

Tests whether removing either of the two targeted regularization components weakens fixed-time reconstruction quality.

- **Metric**: `freeze_psnr` (higher is better)
- **Metric**: `ghosting_error` (lower is better)
- **How to run**: included in `bash scripts/method.sh` and `bash scripts/evaluate.sh`.
