# Overview

- **Paper ID:** 2512.05113
- **Title:** Splannequin: Freezing Monocular Mannequin-Challenge Footage with Dual-Detection Splatting
- **Domain:** Computer Vision / Dynamic Gaussian Splatting
- **TL;DR:** The paper regularizes poorly supervised dynamic Gaussian primitives so monocular near-static videos can be rendered as clean fixed-time scenes.

## Short Summary

The paper studies freeze-time reconstruction from casual Mannequin-Challenge-style videos, where small subject motion makes standard dynamic reconstruction produce ghosting when rendered at one timestamp. Splannequin detects hidden and defective Gaussian states and anchors them to better-supervised states from nearby times. The method is added as training losses, so rendering cost and model architecture are unchanged.

## Key Results

- On the paper's synthetic benchmark, adding Splannequin to 4DGaussians improves PSNR from 28.03 to 28.85 and reduces FVD from 98.93 to 82.73.
- On real videos, the largest reported gain is for D-3DGS + Splannequin: 243.80% CQA average improvement and 339.85% COVER technical improvement.
- A user study reports 96% preference for Splannequin outputs and 80% stronger perceived frozen effect.

---

# Problem Definition

## Research Question

How can a system reconstruct a convincing fixed-time scene from a monocular video in which the camera moves through an almost-frozen environment but people or objects still undergo small unintended motion?

The desired output is a sequence rendered from the input camera poses while holding the scene at one chosen timestamp. A successful system should remove ghosting, blur, double contours, and temporal instability caused by sparse monocular observations.

## Why It Matters

Consumer Mannequin-Challenge-style footage is cheap to capture but imperfect. A reliable fixed-time reconstruction would enable bullet-time and free-viewpoint visual effects without a multi-camera rig.

## Success Criteria

A solution should render fixed-time views that are closer to a static target than a dynamic reconstruction used naively, reduce ghosting artifacts, remain stable across the camera path, and avoid adding inference-time model state or runtime requirements.

---

# Paper's Method

## Key Contribution

Splannequin is an architecture-agnostic regularization framework for dynamic Gaussian splatting. It stabilizes Gaussian primitives that are poorly supervised in monocular Mannequin-Challenge videos so the trained dynamic model can be rendered at a fixed timestamp.

## Approach

The method identifies two problematic primitive states during training:

- Hidden states: the Gaussian center projects outside the current camera frustum.
- Defective states: the center is inside the frustum but its rendered contribution is negligible, so it receives little or no reconstruction gradient.

Hidden states are anchored to recent well-supervised past states. Defective states are anchored to well-supervised future states. The consistency penalty is weighted by an exponential decay in temporal distance, and is added to the normal reconstruction objective. After training, fixed-time rendering is done by evaluating the dynamic Gaussian model once at the selected timestamp and rendering every camera pose with that frozen state.

## Main Claims

The central claim is that dual detection plus temporal anchoring reduces ghosting and blur in fixed-time renderings from imperfect monocular Mannequin-Challenge footage. Secondary claims are that both hidden and defective anchoring are necessary, the method can be added to multiple dynamic Gaussian pipelines, and it adds zero inference overhead.

---

# Evaluation

## Metrics

The paper evaluates fixed-time reconstruction quality with full-reference metrics on synthetic data and no-reference perceptual metrics on real video.

For this reusable reproduction, the runnable benchmark uses a synthetic monocular forward-camera scene with known fixed-time ground truth:

- `freeze_psnr`: PSNR between rendered fixed-time views and static ground truth. Higher is better.
- `ghosting_error`: mean high-frequency residual between rendered and ground-truth views. Lower is better.
- `temporal_error`: mismatch in frame-to-frame variation along the fixed-time camera path. Lower is better.
- `inference_state_values`: number of scalar state values needed at rendering time. Lower is better and checks the zero-overhead claim.

## Baselines and Targets

The baseline is the same dynamic Gaussian representation trained only on diagonal video observations and then rendered at a fixed timestamp. The proposed result should improve fixed-time quality without increasing inference state.

The paper reports synthetic improvements for 4DGaussians + Splannequin over 4DGaussians: PSNR 28.85 vs 28.03, SSIM 0.83 vs 0.81, LPIPS 0.08 vs 0.09, and FVD 82.73 vs 98.93.

## Evaluation Protocol

Train on the full monocular sequence, then render all input camera poses at a single selected timestamp. Evaluation reads saved render outputs from `checkpoints/*/predictions.json` and compares them to `data/synthetic_mannequin.json`. The evaluator does not import or call the method implementation.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 09:50] - none
- Read the paper and identified Splannequin's dual-detection temporal anchoring method.
- Implemented a scaled dynamic Gaussian reproduction, synthetic benchmark generator, evaluation code, scripts, and documentation.
- Submitting first CPU container job to verify the method runs and produces scores.

### [2026-06-01 10:05] - none
- Continued from the prior workspace and audited implementation, evaluator, scripts, and outputs.
- Fixed renderer/evaluator issues before first successful score generation.
- Resubmitting CPU test job to run the complete reproduction pipeline.

### [2026-06-01 10:20] - method_runs
- CPU job ran baseline, Splannequin, and ablation variants end-to-end and wrote real prediction artifacts and scores.
- Splannequin improved fixed-time PSNR and artifact proxy metrics over the unregularized dynamic Gaussian baseline.
- Job failed only at validation because an extra diagnostic score group was not present in `reference.json`; fixed the score contract and resubmitting.

### [2026-06-01 10:35] - core_claim
- CPU test job completed successfully and `python validate.py --compare` passed.
- On the scaled synthetic fixed-time benchmark, Splannequin improved PSNR, LPIPS proxy, FVD proxy, and ghosting error over the unregularized dynamic Gaussian baseline.
- Inference overhead remained zero, matching the paper's constraint claim.
- Secondary ablation behavior was only partially reproduced, so the milestone remains `core_claim`.

</details>


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
