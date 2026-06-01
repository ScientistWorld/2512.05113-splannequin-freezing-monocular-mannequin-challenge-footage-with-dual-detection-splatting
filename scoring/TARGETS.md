# Evaluation Targets

## Primary: synthetic_validation_paper

Tests whether fixed-time renderings are closer to a static target than a dynamic reconstruction rendered naively at one timestamp.

The standard score is the mean over the generated synthetic validation scenes, matching the paper's aggregate synthetic-table style. Per-scene details are diagnostic, not separate optimization targets.

- **Metric**: `psnr` (higher is better)
- **Metric**: `ssim` (higher is better)
- **Metric**: `lpips` (lower is better)
- **Metric**: `fvd` (lower is better)
- **Metric**: `foreground_mae` (lower is better)
- **Metric**: `edge_mae` (lower is better)
- **How to run**: `bash scripts/reproduce.sh` or `bash scripts/evaluate.sh` after writing compatible prediction files.

## Constraint: zero_inference_overhead

Tests whether quality improvements are achieved without increasing the state required at rendering time.

- **Metric**: `inference_overhead` (lower is better; proposed and baseline should match)
- **How to run**: included in `bash scripts/evaluate.sh`.
