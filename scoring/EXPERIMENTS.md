# Experiment Overview

## synthetic_validation_paper

This experiment is the scaled reproduction counterpart of the paper's synthetic fixed-time validation. It asks whether the proposed fixed-time reconstruction system improves over the same dynamic Gaussian representation used without the paper's regularization while preserving inference-time state size.

## ablation_regularizers

This experiment tests whether removing either major regularization component weakens the fixed-time reconstruction, rather than the result being explained only by generic optimization or the synthetic scene setup.

## synthetic_freeze_time

This experiment tests the paper's central fixed-time reconstruction goal on a controlled monocular scene with known static ground truth. It compares a dynamic scene model rendered at a fixed timestamp against variants designed for the same setting, measuring whether fixed-time output is sharper, more stable, and closer to the target scene while keeping rendering-time state fixed.
