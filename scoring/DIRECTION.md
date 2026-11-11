# Research Direction

## Research Problem

The problem is fixed-time reconstruction from monocular near-static video. Each instant is seen by one camera pose, so at any chosen instant much of the scene is unobserved or barely observed. Rendering every camera pose at that instant exposes those states as ghosting, blur, flicker or misplaced content.

## Core Contribution

The paper's contribution is targeted handling of poorly supervised temporal scene states. Future work should improve how such states are identified and stabilized while keeping the ability to choose the freeze instant.

## Approach Scope

In scope are approaches that address fixed-time rendering from sparse monocular temporal supervision: optimization strategy, supervision handling, temporal priors, stability mechanisms, and the deformation model, within the primitive budget.

## Out of Bounds

Static reconstruction is out of bounds because it removes instant selection; it also scores 0 or lower on `instant_selectivity`. So are feed-forward renderers, video diffusion, generic image generators, raising scene capacity, or training priors on external data or extra generated scenes.
