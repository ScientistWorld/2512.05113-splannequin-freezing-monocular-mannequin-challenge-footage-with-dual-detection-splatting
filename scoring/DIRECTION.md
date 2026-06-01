# Research Direction

## Research Problem

The research problem is fixed-time reconstruction from monocular near-static video. The challenge is that a camera observes different parts of the scene at different timestamps, so rendering all views at one timestamp exposes poorly supervised temporal states as ghosting, blur, or unstable geometry.

## Core Contribution

The paper's contribution is targeted handling of poorly supervised time-dependent Gaussian primitive states. Future work should improve how such states are detected, anchored, regularized, or otherwise stabilized while preserving the ability to choose a freeze timestamp.

## Approach Scope

In-scope approaches must remain within dynamic Gaussian-style scene representations and fixed-time rendering. A scientist may change supervision-state detection, anchor selection, confidence weighting, regularization norms, schedules, or optimization strategy.

## Out of Bounds

Replacing the task with static reconstruction is out of bounds because it removes timestamp selection. Replacing Gaussian primitives with a feed-forward renderer, video diffusion model, or generic neural image generator is out of bounds because it sidesteps the paper's representation and supervision problem. Increasing scene density or adding external data for the reported benchmark score is also out of bounds.
