# Research Direction

## Research Problem

The research problem is fixed-time reconstruction from monocular near-static video. The challenge is that a camera observes different parts of the scene at different timestamps, so rendering all views at one timestamp exposes poorly supervised temporal states as ghosting, blur, or unstable geometry.

## Core Contribution

The paper's contribution is targeted handling of poorly supervised temporal scene components. Future work should improve how such components are identified and stabilized while preserving the ability to choose a freeze timestamp.

## Approach Scope

In-scope approaches should address fixed-time rendering from sparse monocular temporal supervision. A scientist may change the optimization strategy, supervision handling, or stability mechanisms.

## Out of Bounds

Replacing the task with static reconstruction is out of bounds because it removes timestamp selection. Replacing the reconstruction system with a feed-forward renderer, video diffusion model, or generic neural image generator is out of bounds because it sidesteps the temporal supervision problem. Increasing scene capacity or adding external data for the reported benchmark score is also out of bounds.
