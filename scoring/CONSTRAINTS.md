# Scientist Constraints

## Task Scope

Submissions must solve fixed-time reconstruction from sparse monocular temporal supervision. Replacing the task with generic image-to-image translation, video diffusion, or direct per-frame regression is out of scope because those systems sidestep the benchmark's temporal reconstruction problem.

## Data

Submissions must train and evaluate on the provided generated validation scenes in `data/` for the reported `synthetic_validation_paper` score. External data, pretrained image models, or privately generated extra scenes may not be used for this score.

## Evaluation

Submissions must write predictions in the same JSON format consumed by `eval/evaluate_freeze.py`. Evaluation must use the provided fixed-time ground truth and must not tune on or alter `gt_freeze`.

## Resource Scale

Submissions may change optimization details but may not increase the scene-capacity budget or image width for the reported score. Improvements should come from better handling of sparse temporal supervision, not from a larger reconstruction budget.

## Compliance Checklist

- [ ] Solves fixed-time reconstruction from the provided monocular sequence.
- [ ] Uses only the provided generated validation scenes in `data/` for the reported score.
- [ ] Does not alter fixed-time ground truth.
- [ ] Does not increase scene-capacity count or render width.
- [ ] Reports scores through `bash scripts/evaluate.sh`.
