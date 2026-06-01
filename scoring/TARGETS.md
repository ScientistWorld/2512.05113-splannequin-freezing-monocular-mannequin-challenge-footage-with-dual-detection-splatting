# Evaluation Targets

## Primary: synthetic_validation_paper

Tests whether fixed-time renderings are closer to a static target than a dynamic reconstruction rendered naively at one timestamp.

- **Metric**: `psnr` (higher is better)
- **Metric**: `ssim` (higher is better)
- **Metric**: `lpips` (lower is better)
- **Metric**: `fvd` (lower is better)
- **How to run**: `bash scripts/reproduce.sh` or `bash scripts/evaluate.sh` after writing compatible prediction files.

## Constraint: zero_inference_overhead

Tests whether quality improvements are achieved without increasing the state required at rendering time.

- **Metric**: `inference_overhead` (lower is better; proposed and baseline should match)
- **How to run**: included in `bash scripts/evaluate.sh`.

## Robustness: synthetic_validation_extra_scenes

Tests whether the same fixed-time reconstruction effect holds on additional procedurally generated forward-camera scenes.

- **Metric**: average `psnr` across extra scenes (higher is better)
- **Metric**: average `lpips` and `fvd` across extra scenes (lower is better)
- **Constraint Metric**: average `inference_overhead` (lower is better; proposed and baseline should match)
- **How to run**: `bash scripts/reproduce.sh`; the evaluator includes this target when `checkpoints/extra_seed_*` outputs exist.
