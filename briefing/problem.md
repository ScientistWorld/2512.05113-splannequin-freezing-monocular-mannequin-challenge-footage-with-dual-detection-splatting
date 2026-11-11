# Problem Definition

## Research Question

How can a system reconstruct a convincing fixed-time ("freeze-time") scene from a monocular video in which the camera moves through an almost-frozen environment, but people or objects still make small unintended motions?

The desired output is a video rendered from every camera pose of the input video while the scene is held at one user-selected instant. Because each instant is observed by only one camera pose, much of the scene is poorly observed at any given instant: parts the camera has already passed, and parts that are still far away or hidden behind other content. Rendering those parts at the chosen instant tends to produce ghosting, blur, double contours and flicker.

## Why It Matters

Consumer Mannequin-Challenge-style footage is cheap to capture but imperfect. A reliable fixed-time reconstruction enables bullet-time and free-viewpoint effects without a multi-camera rig. The user should still be able to choose which instant to freeze, so the subjects' pose and expression at that instant are preserved.

## Success Criteria

A solution should:

- render freeze-time videos close to the true scene state at the chosen instant, from every camera pose;
- keep the ability to select the instant: different instants should show the scene as it was at those instants, not one averaged scene;
- remain stable across the camera path (no flicker or ghosting);
- not increase what has to be stored and rendered at inference time (a fixed budget of scene primitives).
