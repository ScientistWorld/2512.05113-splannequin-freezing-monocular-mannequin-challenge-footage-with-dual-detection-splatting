# Problem Definition

## Research Question

How can a system reconstruct a convincing fixed-time scene from a monocular video in which the camera moves through an almost-frozen environment but people or objects still undergo small unintended motion?

The desired output is a sequence rendered from the input camera poses while holding the scene at one chosen timestamp. A successful system should remove ghosting, blur, double contours, and temporal instability caused by sparse monocular observations.

## Why It Matters

Consumer Mannequin-Challenge-style footage is cheap to capture but imperfect. A reliable fixed-time reconstruction would enable bullet-time and free-viewpoint visual effects without a multi-camera rig.

## Success Criteria

A solution should render fixed-time views that are closer to a static target than a dynamic reconstruction used naively, reduce ghosting artifacts, remain stable across the camera path, and avoid adding inference-time model state or runtime requirements.
