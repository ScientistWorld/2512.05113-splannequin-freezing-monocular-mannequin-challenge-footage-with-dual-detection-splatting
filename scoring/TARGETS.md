# Evaluation Targets

## Primary: synthetic_validation_paper

Freeze-time rendering quality on near-static scenes with small unintended subject motion, scored at private freeze instants on 16 hidden scenes.

- **Metric**: `psnr` (higher is better; primary)
- **Metric**: `ssim` (higher is better)
- **Metric**: `lpips` (perceptual-error proxy, lower is better)
- **Metric**: `fvd` (video-distance proxy, lower is better)
- **Metric**: `edge_mae` (ghosting / double-contour error, lower is better)
- **Constraint metric**: `instant_selectivity` (higher is better). The method must keep user-selectable instants; collapsing to one static scene scores 0.
- **How to run**: write `checkpoints/splannequin/predictions/...` and run `bash scripts/evaluate_train.sh` (visible slice). The hidden test uses the same interface.

## Headroom / limitation: synthetic_large_motion

The same metrics under larger and faster subject motion. This is where the paper's near-static assumption weakens and the reference gains little over its baseline.

- **Metrics**: `psnr` (primary), `ssim`, `lpips`, `fvd`, `edge_mae`, `instant_selectivity`
- **How to run**: as above. Both experiments are scored from the same submission.

## Constraint gate: zero inference overhead

Every submission is a frozen Gaussian state per timestamp with at most 64 primitives, rendered by the evaluator's fixed rasterizer, so inference cost cannot exceed the baseline's. The run also fails on a missing scene file or any non-finite or out-of-range value. A failed run gets the worst value of every metric in both experiments. The gate has no separate metric; failing it is visible as worst values.

## Not scored

The paper's real-video no-reference results (CQA, COVER, IQA; Table 2), its ablation table (Table 3) and its user study cannot be measured offline. Their numbers are kept in `scoring/reference.json` under `_paper_unscored_results`.
