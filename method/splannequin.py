"""Splannequin: dual-detection temporal anchoring for dynamic Gaussian splatting.

Scaled reproduction of Chien et al., "Splannequin: Freezing Monocular
Mannequin-Challenge Footage with Dual-Detection Splatting" (arXiv 2512.05113).

Dynamic model (4DGaussians-style, Sec. 3 / Eq. 1 of the paper): canonical
Gaussians (position, log-scale, opacity, color) plus a HexPlane deformation
field f_theta(mu_k, t) -> (d_mu, d_log_scale): multi-resolution feature planes
(xz spatial, xt and zt spatio-temporal), Hadamard product of bilinear samples,
and a small MLP decoder. Training follows 4DGaussians: a coarse static stage,
then a fine stage with the deformation field and the K-planes TV / time
smoothness / L1-time-plane regularizers. One random training frame (one view
and its timestamp) per iteration, L1 photometric loss.

Splannequin (Sec. 4.1 and 5, paper settings): every `reg_interval` iterations
from `reg_start`, two random view-timestamp pairs (frames a < b) are rendered.
For each primitive and each of the two states:
  * hidden    -- the rasterizer does not see it (behind the near plane or its
                 3-sigma footprint is off the image);
  * defective -- visible, but its reconstruction gradient norm is <= 1e-9
                 (contributions below 1/255 or fully occluded after the
                 transmittance cut-off);
  * well-supervised otherwise.
A hidden state at the later time b is anchored to its well-supervised past
state at a; a defective state at the earlier time a is anchored to its
well-supervised future state at b. The loss is
  lambda * exp(-tau |t - t_ref|) * D(theta_k(t), sg[theta_k(t_ref)]),
summed over instances, with D the squared L2 norm until `l1_start` and the L1
norm afterwards; theta = (position, scale, opacity, color).

After training, fixed-time rendering uses the deformed state at a chosen
timestamp. This script exports that frozen state for every timestamp.

Scale-downs (documented in the gym): 2D scene / 1D image lines instead of 3D /
2D images, a fixed primitive budget without adaptive densification, and the
paper's optimisation schedule otherwise unchanged (30k iterations, anchoring
from 10k, L1 from 20k, lambda = 10, tau = 5, every 10 iterations).
Several scenes are trained in one process as independent batch entries; each
scene's parameters, optimizer state and random schedules are independent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Renderer constants (identical to the benchmark renderer).
FOCAL = 0.78
NEAR = 0.2
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
TRANSMITTANCE_MIN = 1e-4
DILATION_PX2 = 0.3

VARIANTS = {
    "splannequin": {"dynamic": True, "hidden": True, "defective": True},
    "baseline": {"dynamic": True, "hidden": False, "defective": False},
    "static": {"dynamic": False, "hidden": False, "defective": False},
    "no_hidden": {"dynamic": True, "hidden": False, "defective": True},
    "no_defective": {"dynamic": True, "hidden": True, "defective": False},
}


# ----------------------------------------------------------------------------
# Rasterizer


def render(pos, log_scale, opacity, color, cam_x, cam_z, width):
    """Batched 1D Gaussian splatting, one camera per scene.

    pos [B,K,2], log_scale [B,K], opacity [B,K] (activated), color [B,K,3]
    (activated), cam_x/cam_z [B]. Returns rgb [B,W,3] and visibility [B,K]
    (the rasterizer's radii>0 test: in front of the near plane and the
    3-sigma footprint overlaps the image).
    """
    u = torch.arange(width, dtype=pos.dtype, device=pos.device)
    dz = pos[..., 1] - cam_z[:, None]
    in_front = dz > NEAR
    dz_safe = torch.where(in_front, dz, torch.ones_like(dz))
    px = (0.5 + FOCAL * (pos[..., 0] - cam_x[:, None]) / dz_safe) * (width - 1)
    sigma = torch.sqrt((torch.exp(log_scale) * FOCAL * (width - 1) / dz_safe) ** 2 + DILATION_PX2)
    visible = in_front & (px + 3.0 * sigma >= 0.0) & (px - 3.0 * sigma <= width - 1)
    g = torch.exp(-0.5 * ((u[None, None, :] - px[..., None]) / sigma[..., None]) ** 2)
    alpha = torch.clamp(opacity[..., None] * g, max=ALPHA_MAX)
    alpha = torch.where((alpha >= ALPHA_MIN) & in_front[..., None], alpha, torch.zeros_like(alpha))
    depth_key = torch.where(in_front, dz, torch.full_like(dz, float("inf"))).detach()
    order = torch.argsort(depth_key, dim=1, stable=True)  # front to back
    a = torch.gather(alpha, 1, order[..., None].expand(-1, -1, width))
    c = torch.gather(color, 1, order[..., None].expand(-1, -1, 3))
    t_after = torch.cumprod(1.0 - a, dim=1)
    keep = (t_after >= TRANSMITTANCE_MIN).to(a.dtype)  # early termination
    t_before = torch.cat([torch.ones_like(t_after[:, :1]), t_after[:, :-1]], dim=1)
    weight = a * t_before * keep
    rgb = torch.einsum("bkw,bkc->bwc", weight, c)
    return rgb, visible


# ----------------------------------------------------------------------------
# 4DGaussians-style HexPlane deformation field (batched over scenes)


def _sample(plane, gx, gy):
    grid = torch.stack([gx, gy], dim=-1)[:, None]  # [B,1,K,2]
    out = F.grid_sample(plane, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[:, :, 0].transpose(1, 2)  # [B,K,C]


def _linear(x, w, b):
    return torch.bmm(x, w) + b[:, None, :]


def _second_diff_h(t):
    first = t[:, :, 1:, :] - t[:, :, :-1, :]
    second = first[:, :, 1:, :] - first[:, :, :-1, :]
    return second.pow(2).mean(dim=(1, 2, 3))


def _xavier(gens, fan_in, fan_out):
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return torch.stack([torch.empty(fan_in, fan_out).uniform_(-bound, bound, generator=g) for g in gens])


class HexPlaneDeformation(torch.nn.Module):
    def __init__(self, gens, lo, hi, time_res=25, feat_dim=16, base_res=32, multires=(1, 2), hidden=64):
        super().__init__()
        b = len(gens)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.xz = torch.nn.ParameterList()
        self.xt = torch.nn.ParameterList()
        self.zt = torch.nn.ParameterList()
        for m in multires:
            r = base_res * m
            self.xz.append(torch.nn.Parameter(torch.stack([torch.empty(feat_dim, r, r).uniform_(0.1, 0.5, generator=g) for g in gens])))
            self.xt.append(torch.nn.Parameter(torch.ones(b, feat_dim, time_res, r)))
            self.zt.append(torch.nn.Parameter(torch.ones(b, feat_dim, time_res, r)))
        in_dim = feat_dim * len(multires)
        self.w0 = torch.nn.Parameter(_xavier(gens, in_dim, hidden))
        self.b0 = torch.nn.Parameter(torch.zeros(b, hidden))
        self.wp1 = torch.nn.Parameter(_xavier(gens, hidden, hidden))
        self.bp1 = torch.nn.Parameter(torch.zeros(b, hidden))
        self.wp2 = torch.nn.Parameter(torch.zeros(b, hidden, 2))
        self.bp2 = torch.nn.Parameter(torch.zeros(b, 2))
        self.ws1 = torch.nn.Parameter(_xavier(gens, hidden, hidden))
        self.bs1 = torch.nn.Parameter(torch.zeros(b, hidden))
        self.ws2 = torch.nn.Parameter(torch.zeros(b, hidden, 1))
        self.bs2 = torch.nn.Parameter(torch.zeros(b, 1))

    def grid_parameters(self):
        return list(self.xz) + list(self.xt) + list(self.zt)

    def mlp_parameters(self):
        return [self.w0, self.b0, self.wp1, self.bp1, self.wp2, self.bp2, self.ws1, self.bs1, self.ws2, self.bs2]

    def forward(self, pos, t_norm):
        xn = 2.0 * (pos[..., 0] - self.lo[:, None, 0]) / (self.hi[:, None, 0] - self.lo[:, None, 0]) - 1.0
        zn = 2.0 * (pos[..., 1] - self.lo[:, None, 1]) / (self.hi[:, None, 1] - self.lo[:, None, 1]) - 1.0
        tn = (2.0 * t_norm - 1.0)[:, None].expand_as(xn)
        feats = []
        for xz, xt, zt in zip(self.xz, self.xt, self.zt):
            feats.append(_sample(xz, xn, zn) * _sample(xt, xn, tn) * _sample(zt, zn, tn))
        h = _linear(torch.cat(feats, dim=-1), self.w0, self.b0)
        d_pos = _linear(F.relu(_linear(F.relu(h), self.wp1, self.bp1)), self.wp2, self.bp2)
        d_log_scale = _linear(F.relu(_linear(F.relu(h), self.ws1, self.bs1)), self.ws2, self.bs2)[..., 0]
        return d_pos, d_log_scale

    def regularization(self, plane_tv, time_smooth, l1_time):
        total = 0.0
        for xz, xt, zt in zip(self.xz, self.xt, self.zt):
            total = total + plane_tv * _second_diff_h(xz)
            for p in (xt, zt):
                total = total + time_smooth * _second_diff_h(p) + l1_time * (1.0 - p).abs().mean(dim=(1, 2, 3))
        return total  # [B]


class DynamicGaussians(torch.nn.Module):
    def __init__(self, init_pos, init_col, times, gens, args):
        super().__init__()
        b, k, _ = init_pos.shape
        self.register_buffer("times", times)
        self.xyz = torch.nn.Parameter(init_pos.clone())
        d2 = (init_pos[:, :, None, :] - init_pos[:, None, :, :]).pow(2).sum(-1)
        d2 = d2 + torch.eye(k)[None] * 1e10
        knn = torch.topk(d2, 3, dim=-1, largest=False).values.mean(-1).clamp_min(1e-7)
        self.log_scale = torch.nn.Parameter(torch.log(torch.sqrt(knn)))
        self.opacity_logit = torch.nn.Parameter(torch.full((b, k), math.log(0.1 / 0.9)))
        col = init_col.clamp(0.02, 0.98)
        self.color_logit = torch.nn.Parameter(torch.log(col / (1.0 - col)))
        lo = init_pos.min(dim=1).values
        hi = init_pos.max(dim=1).values
        margin = 0.1 * (hi - lo) + 1e-3
        self.field = HexPlaneDeformation(
            gens, lo - margin, hi + margin, args.time_res, args.feat_dim, args.base_res, (1, 2), args.hidden
        )

    def state(self, t_idx, deform):
        pos, log_scale = self.xyz, self.log_scale
        if deform:
            d_pos, d_ls = self.field(self.xyz, self.times[t_idx])
            pos = pos + d_pos
            log_scale = log_scale + d_ls
        return pos, log_scale, torch.sigmoid(self.opacity_logit), torch.sigmoid(self.color_logit)


def theta(state):
    pos, log_scale, opacity, color = state
    return torch.cat([pos, torch.exp(log_scale)[..., None], opacity[..., None], color], dim=-1)


# ----------------------------------------------------------------------------
# Training


def expon_lr(step, lr_init, lr_final, max_steps):
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)


def scene_seed(seed, scene_id):
    return (int(seed) * 1000003 + zlib.crc32(scene_id.encode("utf-8"))) % (2 ** 31 - 1)


class Batch:
    def __init__(self, paths):
        raws = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
        self.scene_ids = [r["scene_id"] for r in raws]
        self.experiments = [r["experiment"] for r in raws]
        self.width = int(raws[0]["width"])
        self.num_frames = int(raws[0]["num_frames"])
        for r in raws:
            if int(r["width"]) != self.width or int(r["num_frames"]) != self.num_frames:
                raise SystemExit("all scenes in one run must share width and frame count")
            if len(r["init_points"]["positions"]) != len(raws[0]["init_points"]["positions"]):
                raise SystemExit("all scenes in one run must share the primitive budget")
        self.frames = torch.tensor([r["frames"] for r in raws], dtype=torch.float32)
        self.cam_x = torch.tensor([r["cam_x"] for r in raws], dtype=torch.float32)
        self.cam_z = torch.tensor([r["cam_z"] for r in raws], dtype=torch.float32)
        self.times = torch.tensor(raws[0]["timestamps"], dtype=torch.float32)
        self.init_pos = torch.tensor([r["init_points"]["positions"] for r in raws], dtype=torch.float32)
        self.init_col = torch.tensor([r["init_points"]["colors"] for r in raws], dtype=torch.float32)
        self.budget = int(raws[0]["primitive_budget"])
        cams = torch.stack([self.cam_x, self.cam_z], dim=-1)  # [B,T,2]
        center = cams.mean(dim=1, keepdim=True)
        self.extent = float(1.1 * (cams - center).norm(dim=-1).max(dim=1).values.mean())
        self.bidx = torch.arange(len(raws))


def detect(model, batch, state, t_idx, retain):
    """Render frames t_idx from their own cameras; return per-primitive gradient norm and visibility."""
    img, vis = render(*state, batch.cam_x[batch.bidx, t_idx], batch.cam_z[batch.bidx, t_idx], batch.width)
    loss = (img - batch.frames[batch.bidx, t_idx]).abs().mean(dim=(1, 2)).sum()
    grads = torch.autograd.grad(loss, list(state), retain_graph=retain, allow_unused=True)
    sq = torch.zeros_like(vis, dtype=torch.float32)
    for g in grads:
        if g is None:
            continue
        sq = sq + (g.pow(2).sum(-1) if g.dim() == 3 else g.pow(2))
    return sq.sqrt().detach(), vis


def anchoring_loss(model, batch, pair, args, cfg, use_l1, stats):
    ta, tb = pair[:, 0], pair[:, 1]  # ta < tb per scene
    sa = model.state(ta, True)
    sb = model.state(tb, True)
    ga, vis_a = detect(model, batch, sa, ta, retain=True)
    gb, vis_b = detect(model, batch, sb, tb, retain=True)
    well_a = vis_a & (ga > args.grad_threshold)
    well_b = vis_b & (gb > args.grad_threshold)
    defective_a = vis_a & ~well_a
    hidden_b = ~vis_b
    phi = torch.exp(-args.tau * (model.times[tb] - model.times[ta]))[:, None]  # [B,1]
    th_a, th_b = theta(sa), theta(sb)

    def disc(x, y):
        d = x - y
        return d.abs().sum(-1) if use_l1 else d.pow(2).sum(-1)

    reg = torch.zeros((), dtype=torch.float32)
    if cfg["hidden"]:
        m = (hidden_b & well_a).float()
        reg = reg + args.lambda_hidden * (phi * disc(th_b, th_a.detach()) * m).sum()
        stats["hidden_anchor_items"] += int(m.sum().item())
    if cfg["defective"]:
        m = (defective_a & well_b).float()
        reg = reg + args.lambda_defective * (phi * disc(th_a, th_b.detach()) * m).sum()
        stats["defective_anchor_items"] += int(m.sum().item())
    stats["regularization_steps"] += 1
    stats["hidden_states_seen"] += int(hidden_b.sum().item())
    stats["defective_states_seen"] += int(defective_a.sum().item())
    return reg


def make_schedules(batch, args):
    t = batch.num_frames
    n_reg = args.iterations // args.reg_interval + 1
    frames, pairs = [], []
    for sid in batch.scene_ids:
        rng = np.random.default_rng(scene_seed(args.seed, sid))
        frames.append(rng.integers(0, t, size=args.iterations))
        a = rng.integers(0, t, size=n_reg)
        b = rng.integers(0, t - 1, size=n_reg)
        b = b + (b >= a)
        pairs.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    return torch.tensor(np.stack(frames), dtype=torch.long), torch.tensor(np.stack(pairs), dtype=torch.long)


def train(batch, args, cfg):
    gens = []
    for sid in batch.scene_ids:
        g = torch.Generator()
        g.manual_seed(scene_seed(args.seed + 7919, sid))
        gens.append(g)
    model = DynamicGaussians(batch.init_pos, batch.init_col, batch.times, gens, args)
    e = batch.extent
    groups = [
        {"params": [model.xyz], "lr": args.position_lr * e, "name": "xyz"},
        {"params": model.field.mlp_parameters(), "lr": args.deformation_lr * e, "name": "deformation"},
        {"params": model.field.grid_parameters(), "lr": args.grid_lr * e, "name": "grid"},
        {"params": [model.color_logit], "lr": args.feature_lr, "name": "color"},
        {"params": [model.opacity_logit], "lr": args.opacity_lr, "name": "opacity"},
        {"params": [model.log_scale], "lr": args.scaling_lr, "name": "scaling"},
    ]
    opt = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    frame_sched, pair_sched = make_schedules(batch, args)
    stats = {
        "regularization_steps": 0,
        "hidden_anchor_items": 0,
        "defective_anchor_items": 0,
        "hidden_states_seen": 0,
        "defective_states_seen": 0,
    }
    start = time.time()
    anchoring = cfg["hidden"] or cfg["defective"]
    for it in range(args.iterations):
        for grp in opt.param_groups:
            if grp["name"] == "xyz":
                grp["lr"] = expon_lr(it, args.position_lr * e, args.position_lr_final * e, args.iterations)
            elif grp["name"] == "deformation":
                grp["lr"] = expon_lr(it, args.deformation_lr * e, args.deformation_lr_final * e, args.iterations)
            elif grp["name"] == "grid":
                grp["lr"] = expon_lr(it, args.grid_lr * e, args.grid_lr_final * e, args.iterations)
        fine = cfg["dynamic"] and it >= args.coarse_iterations
        t_idx = frame_sched[:, it]
        state = model.state(t_idx, fine)
        img, _ = render(*state, batch.cam_x[batch.bidx, t_idx], batch.cam_z[batch.bidx, t_idx], batch.width)
        loss = (img - batch.frames[batch.bidx, t_idx]).abs().mean(dim=(1, 2)).sum()
        if fine:
            loss = loss + model.field.regularization(args.plane_tv_weight, args.time_smoothness_weight, args.l1_time_planes).sum()
        if anchoring and fine and it >= args.reg_start and it % args.reg_interval == 0:
            use_l1 = it >= args.l1_start
            loss = loss + anchoring_loss(model, batch, pair_sched[:, it // args.reg_interval], args, cfg, use_l1, stats)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iteration {it}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % args.log_every == 0 or it == 0:
            print(f"[{args.variant} seed={args.seed}] it {it + 1}/{args.iterations} loss {loss.item():.5f} "
                  f"elapsed {time.time() - start:.1f}s", flush=True)
    stats["train_seconds"] = round(time.time() - start, 1)
    return model, stats


@torch.no_grad()
def frozen_states(model, batch, deform):
    b, t = len(batch.scene_ids), batch.num_frames
    out = {"positions": [], "scale": [], "opacity": [], "color": []}
    for ti in range(t):
        pos, ls, op, col = model.state(torch.full((b,), ti, dtype=torch.long), deform)
        out["positions"].append(pos)
        out["scale"].append(torch.exp(ls))
        out["opacity"].append(op)
        out["color"].append(col)
    return {k: torch.stack(v, dim=1).numpy().astype(np.float64) for k, v in out.items()}  # [B,T,...]


def supervision_masks(model, batch, deform, threshold):
    b, t = len(batch.scene_ids), batch.num_frames
    well = np.zeros((b, t, batch.init_pos.shape[1]), dtype=bool)
    vis = np.zeros_like(well)
    for ti in range(t):
        t_idx = torch.full((b,), ti, dtype=torch.long)
        g, v = detect(model, batch, model.state(t_idx, deform), t_idx, retain=False)
        vis[:, ti] = v.numpy()
        well[:, ti] = (v & (g > threshold)).numpy()
    return well, vis


def snap_to_nearest_supervised(states, well, vis):
    """Post-hoc variant: copy each ill-supervised state from its nearest well-supervised time
    (past for hidden states, future for defective ones, nearest otherwise)."""
    out = {k: v.copy() for k, v in states.items()}
    b, t, k = well.shape
    for bi in range(b):
        for ki in range(k):
            wt = np.flatnonzero(well[bi, :, ki])
            if wt.size == 0:
                continue
            for ti in range(t):
                if well[bi, ti, ki]:
                    continue
                cand = wt[wt < ti] if not vis[bi, ti, ki] else wt[wt > ti]
                if cand.size == 0:
                    cand = wt
                src = int(cand[np.argmin(np.abs(cand - ti))])
                out["positions"][bi, ti, ki] = states["positions"][bi, src, ki]
                out["scale"][bi, ti, ki] = states["scale"][bi, src, ki]
    return out


def write_predictions(out_dir, batch, states, variant, args, stats):
    for bi, (sid, exp) in enumerate(zip(batch.scene_ids, batch.experiments)):
        path = Path(out_dir) / "predictions" / exp / f"{sid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene_id": sid,
            "experiment": exp,
            "variant": variant,
            "seed": int(args.seed),
            "frozen_states": {
                "positions": np.round(states["positions"][bi], 6).tolist(),
                "scale": np.round(np.clip(states["scale"][bi], 1e-6, 10.0), 6).tolist(),
                "opacity": np.round(np.clip(states["opacity"][bi], 0.0, 1.0), 6).tolist(),
                "color": np.round(np.clip(states["color"][bi], 0.0, 1.0), 6).tolist(),
            },
            "training_summary": stats,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", choices=sorted(VARIANTS), default="splannequin")
    p.add_argument("--scenes", nargs="+", required=True, help="scene input JSON files (same width/frames/budget)")
    p.add_argument("--out-dir", required=True, help="method directory; writes predictions/<experiment>/<scene_id>.json")
    p.add_argument("--snap-out-dir", default=None, help="also export the post-hoc nearest-supervised-state variant here")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--log-every", type=int, default=5000)
    # 4DGaussians / 3DGS optimisation settings.
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--coarse-iterations", type=int, default=3000)
    p.add_argument("--position-lr", type=float, default=1.6e-4)
    p.add_argument("--position-lr-final", type=float, default=1.6e-6)
    p.add_argument("--deformation-lr", type=float, default=1.6e-4)
    p.add_argument("--deformation-lr-final", type=float, default=1.6e-6)
    p.add_argument("--grid-lr", type=float, default=1.6e-3)
    p.add_argument("--grid-lr-final", type=float, default=1.6e-5)
    p.add_argument("--feature-lr", type=float, default=2.5e-3)
    p.add_argument("--opacity-lr", type=float, default=0.05)
    p.add_argument("--scaling-lr", type=float, default=5e-3)
    p.add_argument("--feat-dim", type=int, default=16)
    p.add_argument("--base-res", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--time-res", type=int, default=25, help="time-axis resolution of the HexPlane (4DGaussians default 25)")
    p.add_argument("--plane-tv-weight", type=float, default=1e-4)
    p.add_argument("--time-smoothness-weight", type=float, default=1e-2)
    p.add_argument("--l1-time-planes", type=float, default=1e-4)
    # Splannequin settings (paper Sec. 6.1).
    p.add_argument("--reg-start", type=int, default=10000)
    p.add_argument("--l1-start", type=int, default=20000)
    p.add_argument("--reg-interval", type=int, default=10)
    p.add_argument("--lambda-hidden", type=float, default=10.0)
    p.add_argument("--lambda-defective", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--grad-threshold", type=float, default=1e-9)
    args = p.parse_args()

    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    cfg = VARIANTS[args.variant]
    batch = Batch(args.scenes)
    model, stats = train(batch, args, cfg)
    deform = cfg["dynamic"]
    states = frozen_states(model, batch, deform)
    write_predictions(args.out_dir, batch, states, args.variant, args, stats)
    if args.snap_out_dir:
        well, vis = supervision_masks(model, batch, deform, args.grad_threshold)
        snapped = snap_to_nearest_supervised(states, well, vis)
        snap_stats = dict(stats, well_supervised_fraction=float(well.mean()), visible_fraction=float(vis.mean()))
        write_predictions(args.snap_out_dir, batch, snapped, args.variant + "+snap", args, snap_stats)
    print(f"[{args.variant} seed={args.seed}] wrote {len(batch.scene_ids)} scene(s) to {args.out_dir} "
          f"({stats['train_seconds']}s, hidden items {stats['hidden_anchor_items']}, "
          f"defective items {stats['defective_anchor_items']})", flush=True)


if __name__ == "__main__":
    main()
