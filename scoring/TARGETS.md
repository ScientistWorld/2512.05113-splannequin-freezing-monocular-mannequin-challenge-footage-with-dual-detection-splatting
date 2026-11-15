# Evaluation Targets

## Primary: synthetic_validation_paper

Freeze-time rendering quality on near-static scenes with small unintended subject motion, scored at private freeze instants on 40 hidden scenes.

- **Metric**: `psnr` (higher is better; primary)
- **Metric**: `ssim` (higher is better)
- **Metric**: `lpips` (perceptual-error proxy, lower is better)
- **Metric**: `fvd` (video-distance proxy, lower is better)
- **Metric**: `edge_mae` (ghosting / double-contour error, lower is better)
- **Constraint metric**: `instant_selectivity` (higher is better). The method must keep user-selectable instants. Rendering one scene for every instant scores at most 0 here, and a frozen state that is identical at every timestamp fails the run (see the gate below).
- **How to run**: write `checkpoints/splannequin/predictions/...` and run `bash scripts/evaluate_train.sh` (visible slice). The hidden test uses the same interface.

## Headroom / limitation: synthetic_large_motion

The same metrics under larger and faster subject motion, on 16 hidden scenes. This is where the paper's near-static assumption weakens: the scores of every method drop by several dB, leaving room above the reference.

- **Metrics**: `psnr` (primary), `ssim`, `lpips`, `fvd`, `edge_mae`, `instant_selectivity`
- **How to run**: as above. Both experiments are scored from the same submission.

## Constraint gate: zero inference overhead and instant selection

Every submission is a frozen Gaussian state per timestamp with at most 64 primitives, rendered by the evaluator's fixed rasterizer, so inference cost cannot exceed the baseline's. The run also fails on a missing scene file, any non-finite or out-of-range value, or a scene whose frozen state is identical at every timestamp (it cannot select an instant). A failed run gets the worst value of every metric in both experiments. The gate has no separate metric; failing it is visible as worst values.

## Not scored

The paper's real-video no-reference results (CQA, COVER, IQA; Table 2), its ablation table (Table 3) and its user study cannot be measured offline. Their numbers are kept in `scoring/reference.json` under `_paper_unscored_results`.
