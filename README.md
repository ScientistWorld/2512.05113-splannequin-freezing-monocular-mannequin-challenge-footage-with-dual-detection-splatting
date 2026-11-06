# Splannequin: Freezing Monocular Mannequin-Challenge Footage with Dual-Detection Splatting

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2512.05113"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2512.05113-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2512.05113](https://arxiv.org/abs/2512.05113) |
| **Domain** | Computer Vision / Dynamic Gaussian Splatting |
| **TL;DR** | The paper regularizes poorly supervised dynamic Gaussian primitives so monocular near-static videos can be rendered as clean fixed-time scenes. |

## Summary

The paper studies freeze-time reconstruction from casual Mannequin-Challenge-style videos, where small subject motion makes standard dynamic reconstruction produce ghosting when rendered at one timestamp. Splannequin detects hidden and defective Gaussian states and anchors them to better-supervised states from nearby times. The method is added as training losses, so rendering cost and model architecture are unchanged.

## Problem

How can a system reconstruct a convincing fixed-time scene from a monocular video in which the camera moves through an almost-frozen environment but people or objects still undergo small unintended motion?

The desired output is a sequence rendered from the input camera poses while holding the scene at one chosen timestamp. A successful system should remove ghosting, blur, double contours, and temporal instability caused by sparse monocular observations.

## The paper's method

Splannequin is an architecture-agnostic regularization framework for dynamic Gaussian splatting. It stabilizes Gaussian primitives that are poorly supervised in monocular Mannequin-Challenge videos so the trained dynamic model can be rendered at a fixed timestamp.

## Key results of the paper

- On the paper's synthetic benchmark, adding Splannequin to 4DGaussians improves PSNR from 28.03 to 28.85 and reduces FVD from 98.93 to 82.73.
- On real videos, the largest reported gain is for D-3DGS + Splannequin: 243.80% CQA average improvement and 339.85% COVER technical improvement.
- A user study reports 96% preference for Splannequin outputs and 80% stronger perceived frozen effect.

## Run it

```bash
pip install sciworld
sw research 2512.05113 --mode innovate --agent claude-code
sw evaluate 2512.05113
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

