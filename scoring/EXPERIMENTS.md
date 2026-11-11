# Experiment Overview

Both experiments ask the same question: given a monocular video of an almost-frozen scene, can a method render the scene as it was at a chosen instant from every camera pose of the video? Ghosting, blur, flicker and loss of the instant's specific content all count against it. Scores come from rendering the submitted frozen scene states with a fixed rasterizer and comparing them with the true scene at private freeze instants, on 16 hidden scenes per experiment. The validity gate is that a method stays within a fixed 64-primitive budget (no extra inference-time state) and every value is valid. A method that fails the gate receives the worst value of every metric.

## synthetic_validation_paper (weight 0.7)

This is the scaled counterpart of the paper's synthetic validation (paper Table 1). Subjects make small unintended motions, as in Mannequin-Challenge footage. It establishes whether fixed-time renderings get closer to the true state at the chosen instant (`psnr` primary, plus `ssim`, the `lpips` proxy, the `fvd` proxy and `edge_mae`). It also checks that this does not come at the cost of instant selection (`instant_selectivity`): a method that renders one averaged scene for every instant scores 0 there.

## synthetic_large_motion (weight 0.3)

This uses the same protocol and metrics, with larger and faster subject motion. The paper names this as the regime where near-static assumptions break down and leaves it to future work. It measures how much of the quality and instant-selection ability survives when motion is no longer negligible, and leaves room above the reference.
