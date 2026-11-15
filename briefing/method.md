# Paper's Method

## Key Contribution

Splannequin is an architecture-agnostic regularization framework for dynamic Gaussian splatting. It stabilizes Gaussian primitives that are poorly supervised in monocular Mannequin-Challenge videos, so that the trained dynamic model can be rendered at a fixed timestamp without ghosting.

## Approach

A dynamic Gaussian model has canonical Gaussians plus a deformation field f(μ_k, t) giving per-time offsets. During training, the method classifies each primitive state (k, t):

- Hidden: not seen by the rasterizer at time t. Its center is behind the near plane or its footprint is off-image, typically because the camera has passed it.
- Defective: inside the frustum, but its reconstruction gradient norm is ≤ 1e-9 (negligible or fully occluded contribution).
- Well-supervised: otherwise.

Every 10 iterations, two random view–timestamp pairs are sampled. A hidden state is anchored to its well-supervised past state, and a defective state to its well-supervised future state. The loss is λ·exp(−τ|t − t_ref|)·D(θ_k(t), θ_k(t_ref)), summed over instances. Here θ is (position, covariance/scale, opacity, color), and D is squared L2 at first and L1 later. Paper settings: 30k iterations, anchoring from 10k, L1 from 20k, λ_hidden = λ_defective = 10, τ = 5. Fixed-time rendering evaluates the deformation once at t* and renders every camera pose.

## This Reproduction

`method/splannequin.py` implements:

- A 4DGaussians-style model: a HexPlane with xz, xt and zt planes, multires (1, 2), time resolution 25, and an MLP decoder producing position and log-scale offsets.
- 4DGaussians-style training:
  - a 3k-iteration coarse static stage, then the K-planes TV, time-smoothness and L1-time regularizers;
  - one frame per iteration, popped from a view stack without replacement;
  - L1 photometric loss and 3DGS learning rates;
  - standard adaptive density control capped at the 64-primitive budget: every 100 iterations from 500 to 15k, primitives with opacity < 0.005 are pruned and their slots refilled by splitting or cloning the highest-gradient primitives; opacity is reset every 3k iterations until 15k.
- Splannequin with the paper's settings, with two notes:
  - The defective test applies the paper's 1e-9 gradient threshold to the loss normalised like the paper's 640×360 RGB frames (`--grad-loss-values 691200`), so it keeps the paper's per-pixel meaning on a 128-pixel line.
  - The anchor is held fixed: the anchoring gradient reaches the deformation of the ill-supervised state but not the canonical parameters it shares with the anchor. `--anchor-canonical-grad` restores the other reading.

Variants (`--variant`): `splannequin` (reference), `baseline` (4DGaussians without anchoring), `static` (3DGS-style, no deformation; it fails the evaluator's instant-selection gate), `no_hidden`, `no_defective`. `--snap-out-dir` additionally exports a post-hoc variant that copies each ill-supervised state from its nearest well-supervised time.

The scale-downs are 2D scenes with 1D image lines, a 64-primitive cap on densification, and several scenes trained as independent batch entries.

## Main Claims

Dual detection plus temporal anchoring reduces ghosting and blur in fixed-time renderings and improves full-reference quality over the dynamic baseline on synthetic scenes (Table 1). Both anchors are claimed to be necessary (Table 3). The method adds zero inference overhead. Unlike static 3DGS, it keeps user-selectable instants (Sec. 6.4, Table 4).

In this gym (hidden test, see `HARDENING.md`):
- **Table 1 direction reproduces in both experiments.** Splannequin beats 4DGaussians by +1.0 dB PSNR (near-static) and +0.9 dB (large motion), and is better on SSIM, the LPIPS proxy, the FVD proxy and edge error.
- **Hidden anchoring is necessary** (−0.85 dB without it).
- **Defective anchoring helps only a little.** It goes the paper's direction on quality, mostly on early freeze instants, but is small (≈ −0.02 to −0.06 dB PSNR, −0.15 dB on early instants). The paper's finding that it matters more than hidden anchoring is not reproduced.
- **Static reconstruction matches the method on some quality metrics but loses instant selection.** It is better on SSIM, LPIPS and FVD in the near-static experiment (as in Table 4) but loses on PSNR, under large motion, and on instant selection.
