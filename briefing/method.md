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
