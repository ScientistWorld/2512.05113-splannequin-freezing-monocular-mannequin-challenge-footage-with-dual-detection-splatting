# Scientist Constraints

## Task Scope

Submissions must solve fixed-time reconstruction from the provided monocular inputs. Each scene is reconstructed from its own frames, camera poses, timestamps and initial point cloud, and the output is a frozen Gaussian state per timestamp. Replacing the task with image-to-image translation, video diffusion, or regression of the renders themselves is out of scope.

## Data

Use only the provided scenes. The public training scenes (`data/train`, answers included) may be used for development and hyperparameter selection. Test scenes (`data/test`) may only be used as reconstruction inputs. Do not train priors on additional scenes produced with the public generator, and do not use external data or pretrained image models.

## Evaluation

Write predictions in the format in `briefing/evaluation.md` to `checkpoints/splannequin/predictions/`, covering every train and test scene of both experiments. Do not modify `eval/`, `data/scenes.py`, `data/train` or `data/test`. Scores are computed only by the evaluators from the submitted frozen states.

## Resource Scale

At most 64 primitives per frozen state (enforced), with the fixed renderer and image width. Improvements must come from better handling of sparse temporal supervision, not from a larger reconstruction budget.

## Compliance Checklist

- [ ] Reconstructs each scene from its own inputs only.
- [ ] Produces one valid frozen state per timestamp, with ≤ 64 primitives, for every train and test scene.
- [ ] Does not modify evaluators, scenes or the generator.
- [ ] Reports visible scores with `bash scripts/evaluate_train.sh`.
