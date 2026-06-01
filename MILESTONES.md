# Reproduction Milestones

**Current: core_claim**

## Progress Log

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

## Stop Justification
