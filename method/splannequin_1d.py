"""Scaled Splannequin reproduction on differentiable 1D Gaussian splatting.

This preserves the paper's core computation: dynamic Gaussian states are
trained from monocular diagonal observations; hidden and defective states are
detected from visibility/contribution and temporally anchored to supervised
past/future states before fixed-time rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_scene(path: str) -> dict[str, torch.Tensor | int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = ["frames", "gt_freeze", "base_positions", "colors", "opacity", "scale", "cam_z"]
    out = {k: torch.tensor(raw[k], dtype=torch.float32) for k in keys}
    out["freeze_index"] = int(raw["freeze_index"])
    out["width"] = int(raw["width"])
    return out


def render(states, colors, opacity_logits, log_scale, cam_z, width):
    device = states.device
    x = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    dz = states[:, 1] - cam_z
    px = (0.5 + 0.78 * states[:, 0] / dz.clamp_min(1e-3)) * (width - 1)
    visible = (dz > 0.18) & (px > -0.15 * width) & (px < 1.15 * width)
    sigma = (log_scale.exp() * width / dz.clamp_min(0.35)).clamp(0.8, 7.5)[:, None]
    opacity = torch.sigmoid(opacity_logits)
    gauss = torch.exp(-0.5 * ((x - px[:, None]) / sigma) ** 2) * visible[:, None].float()
    raw_alpha = (opacity[:, None] * gauss).clamp(0.0, 0.97)
    order = torch.argsort(dz, descending=True)
    trans = torch.ones((width,), device=device)
    rgb = torch.full((width, 3), 0.02, device=device)
    contrib = torch.zeros((states.shape[0],), device=device)
    color = torch.sigmoid(colors)
    for k in order:
        alpha = raw_alpha[k] * trans
        contrib[k] = alpha.sum()
        rgb = rgb + alpha[:, None] * color[k]
        trans = trans * (1.0 - alpha).clamp_min(0.0)
    return rgb.clamp(0.0, 1.0), visible, contrib


def nearest_anchor_masks(well: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t_count, k_count = well.shape
    past = torch.full((t_count, k_count), -1, dtype=torch.long, device=well.device)
    future = torch.full((t_count, k_count), -1, dtype=torch.long, device=well.device)
    last = torch.full((k_count,), -1, dtype=torch.long, device=well.device)
    for t in range(t_count):
        past[t] = last
        last = torch.where(well[t], torch.full_like(last, t), last)
    nxt = torch.full((k_count,), -1, dtype=torch.long, device=well.device)
    for t in range(t_count - 1, -1, -1):
        future[t] = nxt
        nxt = torch.where(well[t], torch.full_like(nxt, t), nxt)
    return past, future


def train(args):
    torch.manual_seed(args.seed)
    scene = load_scene(args.data)
    frames = scene["frames"]
    cam_z = scene["cam_z"]
    t_count, width, _ = frames.shape
    base_pos = scene["base_positions"]
    k_count = base_pos.shape[0]
    device = torch.device("cpu")
    frames, cam_z, base_pos = frames.to(device), cam_z.to(device), base_pos.to(device)
    state = torch.nn.Parameter(base_pos[None].repeat(t_count, 1, 1) + args.init_noise * torch.randn(t_count, k_count, 2))
    colors = torch.nn.Parameter(torch.logit(scene["colors"].clamp(1e-3, 0.997).to(device)))
    opacity = torch.nn.Parameter(torch.logit(scene["opacity"].clamp(1e-3, 0.95).to(device)))
    log_scale = torch.nn.Parameter(scene["scale"].to(device).log())
    opt = torch.optim.Adam([state, colors, opacity, log_scale], lr=args.lr)

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        recon = torch.zeros((), device=device)
        visible_rows, contrib_rows = [], []
        for t in range(t_count):
            img, vis, con = render(state[t], colors, opacity, log_scale, cam_z[t], width)
            recon = recon + F.mse_loss(img, frames[t])
            visible_rows.append(vis.detach())
            contrib_rows.append(con.detach())
        loss = recon / t_count
        if args.variant != "baseline" and step >= args.reg_start:
            visible = torch.stack(visible_rows)
            contrib = torch.stack(contrib_rows)
            well = visible & (contrib > args.defective_threshold)
            hidden = ~visible
            defective = visible & ~well
            past, future = nearest_anchor_masks(well)
            reg = torch.zeros((), device=device)
            count = torch.zeros((), device=device)
            for t in range(t_count):
                if args.variant != "no_hidden":
                    mask = hidden[t] & (past[t] >= 0)
                    if mask.any():
                        idx = past[t, mask]
                        d = state[t, mask] - state[idx, torch.arange(k_count, device=device)[mask]]
                        weight = torch.exp(-args.tau * (t - idx).float() / max(1, t_count - 1))[:, None]
                        reg = reg + (weight * d.abs()).mean()
                        count = count + 1
                if args.variant != "no_defective":
                    mask = defective[t] & (future[t] >= 0)
                    if mask.any():
                        idx = future[t, mask]
                        d = state[t, mask] - state[idx, torch.arange(k_count, device=device)[mask]]
                        weight = torch.exp(-args.tau * (idx - t).float() / max(1, t_count - 1))[:, None]
                        reg = reg + (weight * d.abs()).mean()
                        count = count + 1
            if count.item() > 0:
                loss = loss + args.lambda_reg * reg / count
        loss.backward()
        opt.step()

    freeze = int(scene["freeze_index"])
    renders = []
    with torch.no_grad():
        for t in range(t_count):
            img, _, _ = render(state[freeze], colors, opacity, log_scale, cam_z[t], width)
            renders.append(img.cpu())
    out = {
        "variant": args.variant,
        "freeze_index": freeze,
        "pred_freeze": torch.stack(renders).tolist(),
        "num_primitives": int(k_count),
        "trained_state_values": int(state.numel()),
        "inference_state_values": int(state[freeze].numel() + colors.numel() + opacity.numel() + log_scale.numel()),
        "steps": int(args.steps),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_mannequin.json")
    parser.add_argument("--out", default="checkpoints/splannequin/predictions.json")
    parser.add_argument("--variant", choices=["baseline", "splannequin", "no_hidden", "no_defective"], default="splannequin")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--reg-start", type=int, default=250)
    parser.add_argument("--lambda-reg", type=float, default=4.0)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--defective-threshold", type=float, default=0.25)
    parser.add_argument("--init-noise", type=float, default=0.13)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=7)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
