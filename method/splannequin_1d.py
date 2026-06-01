"""Scaled Splannequin reproduction on differentiable 1D Gaussian splatting.

This preserves the paper's core computation at small scale: dynamic Gaussian
states are trained from monocular diagonal observations; hidden and defective
states are detected from visibility and near-zero reconstruction gradients; a
randomly sampled well-supervised past/future reference anchors ill-supervised
states before fixed-time rendering.
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


def anchored_discrepancy(
    state: torch.Tensor,
    color: torch.Tensor,
    opacity: torch.Tensor,
    log_scale: torch.Tensor,
    t: int,
    ref: int,
    mask: torch.Tensor,
    use_l1: bool,
) -> torch.Tensor:
    """Discrepancy over the time-varying Gaussian parameters named in the paper."""
    lhs = torch.cat(
        [
            state[t, mask].reshape(-1, 2),
            color[t, mask].reshape(-1, 3),
            opacity[t, mask].reshape(-1, 1),
            log_scale[t, mask].reshape(-1, 1),
        ],
        dim=1,
    )
    rhs = torch.cat(
        [
            state[ref, mask].reshape(-1, 2),
            color[ref, mask].reshape(-1, 3),
            opacity[ref, mask].reshape(-1, 1),
            log_scale[ref, mask].reshape(-1, 1),
        ],
        dim=1,
    )
    diff = lhs - rhs
    if use_l1:
        return diff.abs().mean()
    return diff.square().mean()


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
    state = torch.nn.Parameter(
        base_pos[None].repeat(t_count, 1, 1) + args.init_noise * torch.randn(t_count, k_count, 2)
    )
    base_color = torch.logit(scene["colors"].clamp(1e-3, 0.997).to(device))
    base_opacity = torch.logit(scene["opacity"].clamp(1e-3, 0.95).to(device))
    base_scale = scene["scale"].to(device).log()
    color_state = torch.nn.Parameter(
        base_color[None].repeat(t_count, 1, 1) + 0.10 * args.init_noise * torch.randn(t_count, k_count, 3)
    )
    opacity_state = torch.nn.Parameter(
        base_opacity[None].repeat(t_count, 1) + 0.10 * args.init_noise * torch.randn(t_count, k_count)
    )
    log_scale_state = torch.nn.Parameter(
        base_scale[None].repeat(t_count, 1) + 0.10 * args.init_noise * torch.randn(t_count, k_count)
    )
    opt = torch.optim.Adam([state, color_state, opacity_state, log_scale_state], lr=args.lr)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 1009)
    summary = {
        "regularization_updates": 0,
        "hidden_anchor_terms": 0,
        "defective_anchor_terms": 0,
        "hidden_anchor_items": 0,
        "defective_anchor_items": 0,
        "regularization_loss_sum": 0.0,
    }

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        recon = torch.zeros((), device=device)
        visible_rows, grad_rows = [], []
        for t in range(t_count):
            img, vis, _ = render(state[t], color_state[t], opacity_state[t], log_scale_state[t], cam_z[t], width)
            frame_loss = F.mse_loss(img, frames[t])
            recon = recon + frame_loss
            visible_rows.append(vis.detach())
            if args.variant != "baseline" and step >= args.reg_start and step % args.reg_interval == 0:
                grads = torch.autograd.grad(
                    frame_loss,
                    (state, color_state, opacity_state, log_scale_state),
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_norm = grads[0][t].detach().norm(dim=1)
                grad_norm = grad_norm + grads[1][t].detach().norm(dim=1)
                grad_norm = grad_norm + grads[2][t].detach().abs()
                grad_norm = grad_norm + grads[3][t].detach().abs()
            else:
                grad_norm = torch.full((k_count,), float("inf"), device=device)
            grad_rows.append(grad_norm)
        loss = recon / t_count
        if args.variant != "baseline" and step >= args.reg_start and step % args.reg_interval == 0:
            visible = torch.stack(visible_rows)
            grad_norms = torch.stack(grad_rows)
            well = visible & (grad_norms > args.defective_gradient_threshold)
            hidden = ~visible
            defective = visible & ~well
            reg = torch.zeros((), device=device)
            count = torch.zeros((), device=device)
            use_l1 = step >= args.l1_start
            for t in range(t_count):
                for _ in range(args.anchor_samples):
                    ref = int(torch.randint(0, t_count, (1,), generator=rng, device=device).item())
                    if args.variant != "no_hidden" and ref < t:
                        mask = hidden[t] & well[ref]
                        if mask.any():
                            d = anchored_discrepancy(
                                state, color_state, opacity_state, log_scale_state, t, ref, mask, use_l1
                            )
                            weight = torch.exp(torch.tensor(-args.tau * (t - ref) / max(1, t_count - 1), device=device))
                            reg = reg + weight * d
                            count = count + 1
                            summary["hidden_anchor_terms"] += 1
                            summary["hidden_anchor_items"] += int(mask.sum().item())
                    if args.variant != "no_defective" and ref > t:
                        mask = defective[t] & well[ref]
                        if mask.any():
                            d = anchored_discrepancy(
                                state, color_state, opacity_state, log_scale_state, t, ref, mask, use_l1
                            )
                            weight = torch.exp(torch.tensor(-args.tau * (ref - t) / max(1, t_count - 1), device=device))
                            reg = reg + weight * d
                            count = count + 1
                            summary["defective_anchor_terms"] += 1
                            summary["defective_anchor_items"] += int(mask.sum().item())
            if count.item() > 0:
                summary["regularization_updates"] += 1
                summary["regularization_loss_sum"] += float((reg / count).detach().item())
                loss = loss + args.lambda_reg * reg / count
        loss.backward()
        opt.step()

    freeze = int(scene["freeze_index"])
    renders = []
    with torch.no_grad():
        for t in range(t_count):
            img, _, _ = render(
                state[freeze],
                color_state[freeze],
                opacity_state[freeze],
                log_scale_state[freeze],
                cam_z[t],
                width,
            )
            renders.append(img.cpu())
    inference_state_values = (
        state[freeze].numel()
        + color_state[freeze].numel()
        + opacity_state[freeze].numel()
        + log_scale_state[freeze].numel()
    )
    out = {
        "variant": args.variant,
        "freeze_index": freeze,
        "pred_freeze": torch.stack(renders).tolist(),
        "num_primitives": int(k_count),
        "trained_state_values": int(state.numel() + color_state.numel() + opacity_state.numel() + log_scale_state.numel()),
        "inference_state_values": int(inference_state_values),
        "steps": int(args.steps),
        "training_summary": summary,
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
    parser.add_argument("--reg-start", type=int, default=550)
    parser.add_argument("--lambda-reg", type=float, default=0.75)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--defective-gradient-threshold", type=float, default=1e-9)
    parser.add_argument("--anchor-samples", type=int, default=1)
    parser.add_argument("--reg-interval", type=int, default=10)
    parser.add_argument("--l1-start", type=int, default=750)
    parser.add_argument("--init-noise", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=7)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
