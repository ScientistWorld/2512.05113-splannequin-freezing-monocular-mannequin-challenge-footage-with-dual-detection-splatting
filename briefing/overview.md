# Overview

- **Paper ID:** 2512.05113
- **Title:** Splannequin: Freezing Monocular Mannequin-Challenge Footage with Dual-Detection Splatting
- **Domain:** Computer Vision / Dynamic Gaussian Splatting
- **TL;DR:** The paper regularizes poorly supervised dynamic Gaussian primitives so monocular near-static videos can be rendered as clean fixed-time scenes.

## Short Summary

The paper studies freeze-time reconstruction from casual Mannequin-Challenge-style videos. Small subject motion makes standard dynamic reconstruction produce ghosting when it is rendered at one timestamp. Splannequin detects hidden and defective Gaussian states and anchors them to better-supervised states from nearby times. The method is added as training losses, so rendering cost and model architecture are unchanged.

## Key Results (paper)

- On the paper's synthetic benchmark, adding Splannequin to 4DGaussians improves PSNR from 28.03 to 28.85 and reduces FVD from 98.93 to 82.73.
- On real videos, the largest reported gain is for D-3DGS + Splannequin: 243.80% CQA average improvement and 339.85% COVER technical improvement.
- A user study reports 96% preference for Splannequin outputs, and 80% judged its results more "perfectly frozen" than the original captures.

## This Gym

The gym is a scaled synthetic benchmark: 2D scenes with 1D image lines, a forward-moving camera, and moving subjects. Its hidden test scores frozen Gaussian states on private scenes at private freeze instants, in a near-static experiment and a larger-motion stress test. See `briefing/evaluation.md` and `HARDENING.md`.
