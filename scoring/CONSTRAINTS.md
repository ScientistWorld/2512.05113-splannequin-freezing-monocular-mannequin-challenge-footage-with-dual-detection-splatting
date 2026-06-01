# Scientist Constraints

## Model Family

Submissions must use Gaussian primitive scene representations with time-dependent primitive state. The benchmark is about fixed-time reconstruction from sparse monocular temporal supervision, so replacing the representation with a generic image-to-image model, video diffusion model, or direct per-frame regression system is out of scope.

## Data

Submissions must train and evaluate on `data/synthetic_mannequin.json` for the reported `synthetic_freeze_time` score. External data, pretrained image models, or additional generated scenes may not be used for this score.

## Evaluation

Submissions must write predictions in the same JSON format consumed by `eval/evaluate_freeze.py`. Evaluation must use the provided fixed-time ground truth and must not tune on or alter `gt_freeze`.

## Resource Scale

Submissions may change optimization details but may not increase the number of Gaussian primitives or the image width for the reported score. Improvements should come from better handling of ill-supervised temporal states, not from a denser scene representation.

## Compliance Checklist

- [ ] Uses time-dependent Gaussian primitives.
- [ ] Uses only `data/synthetic_mannequin.json` for the reported score.
- [ ] Does not alter fixed-time ground truth.
- [ ] Does not increase primitive count or render width.
- [ ] Reports scores through `bash scripts/evaluate.sh`.
