"""Hardening audit driver (iteration 11).

  python redteam/hardening/audit.py train    # train reference, baselines and variants (resumable)
  python redteam/hardening/audit.py cases    # build every case submission and score it with the hidden test
  python redteam/hardening/audit.py summary  # paired comparisons, seed spread, illustrative rank deltas

Training outputs: the reference method writes to checkpoints/splannequin (the
gym's reference artifacts); every other variant writes to
tmp/hardening/methods/<name>. Each case is scored by running the unmodified
hidden evaluator (eval/test/evaluate.py, the program scripts/evaluate_test.sh
runs) on a case-specific checkpoints directory in which the case's
submission occupies the method slot.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "tmp" / "hardening"
METHODS = HW / "methods"
CASES = HW / "cases"
LOGS = HW / "logs"
DONE = HW / "done"
RESULTS = ROOT / "redteam" / "hardening" / "results"
REF_DIR = ROOT / "checkpoints" / "splannequin"
EXPERIMENTS = ("synthetic_validation_paper", "synthetic_large_motion")
SLOT = "4dgaussians_splannequin"


def scene_set(split, experiment):
    return sorted((ROOT / "data" / split / experiment).glob("*.json"))


def tasks():
    t = []

    def add(name, variant, seed, extra, split, out_dir, snap_dir=None):
        for exp, tag in (("synthetic_validation_paper", "paper"), ("synthetic_large_motion", "large")):
            t.append(
                {
                    "name": f"{name}_{split}_{tag}",
                    "variant": variant,
                    "seed": seed,
                    "extra": extra,
                    "scenes": [str(p) for p in scene_set(split, exp)],
                    "out_dir": str(out_dir),
                    "snap_dir": str(snap_dir) if snap_dir else None,
                }
            )

    add("reference", "splannequin", 0, [], "test", REF_DIR, METHODS / "splannequin_snap")
    add("baseline", "baseline", 0, [], "test", METHODS / "baseline", METHODS / "baseline_snap")
    add("static", "static", 0, [], "test", METHODS / "static")
    add("reference", "splannequin", 0, [], "train", REF_DIR, METHODS / "splannequin_snap")
    add("seed1", "splannequin", 1, [], "test", METHODS / "splannequin_seed1")
    add("seed2", "splannequin", 2, [], "test", METHODS / "splannequin_seed2")
    add("no_hidden", "no_hidden", 0, [], "test", METHODS / "no_hidden")
    add("no_defective", "no_defective", 0, [], "test", METHODS / "no_defective")
    add("tau20", "splannequin", 0, ["--tau", "20"], "test", METHODS / "splannequin_tau20")
    add("lambda1", "splannequin", 0, ["--lambda-hidden", "1", "--lambda-defective", "1"], "test", METHODS / "splannequin_lambda1")
    add("baseline", "baseline", 0, [], "train", METHODS / "baseline", METHODS / "baseline_snap")
    add("static", "static", 0, [], "train", METHODS / "static")
    return t


def run_task(task):
    marker = DONE / f"{task['name']}.json"
    if marker.exists() and json.loads(marker.read_text()).get("returncode") == 0:
        return task["name"], 0, 0.0, True
    cmd = [
        sys.executable,
        str(ROOT / "method" / "splannequin.py"),
        "--variant",
        task["variant"],
        "--seed",
        str(task["seed"]),
        "--out-dir",
        task["out_dir"],
        *task["extra"],
        "--scenes",
        *task["scenes"],
    ]
    if task["snap_dir"]:
        cmd[2:2] = ["--snap-out-dir", task["snap_dir"]]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.time()
    with open(LOGS / f"{task['name']}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    secs = time.time() - start
    if rc == 0:
        marker.write_text(json.dumps({"returncode": rc, "seconds": round(secs, 1), "cmd": cmd[1:]}))
    print(f"[train] {task['name']}: rc={rc} {secs:.0f}s", flush=True)
    return task["name"], rc, secs, False


def cmd_train():
    for d in (METHODS, LOGS, DONE):
        d.mkdir(parents=True, exist_ok=True)
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    workers = int(os.environ.get("HARDEN_WORKERS", max(1, ncpu)))
    tl = tasks()
    print(f"[train] {len(tl)} tasks on {workers} workers (cpus={ncpu})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run_task, tl))
    failed = [n for n, rc, _, _ in results if rc != 0]
    print(f"[train] finished; failed: {failed}", flush=True)
    return 1 if failed else 0


# ----------------------------------------------------------------------------- cases


def _load_scenes_module():
    spec = importlib.util.spec_from_file_location("bench_scenes", ROOT / "data" / "scenes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scene_ids(experiment):
    return [p.stem for p in scene_set("test", experiment)]


def write_pred(method_dir, experiment, scene_id, payload):
    path = Path(method_dir) / "predictions" / experiment / f"{scene_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def copy_modified(src_dir, dst_dir, fn, experiments=EXPERIMENTS, drop=None):
    for exp in experiments:
        for sid in test_scene_ids(exp):
            if drop and (exp, sid) in drop:
                continue
            payload = json.loads((Path(src_dir) / "predictions" / exp / f"{sid}.json").read_text())
            write_pred(dst_dir, exp, sid, fn(copy.deepcopy(payload)))


def states_payload(sid, positions, scale, opacity, color):
    return {
        "scene_id": sid,
        "frozen_states": {
            "positions": np.round(positions, 6).tolist(),
            "scale": np.round(scale, 6).tolist(),
            "opacity": np.round(opacity, 6).tolist(),
            "color": np.round(color, 6).tolist(),
        },
    }


def answers_payload(ans, jitter=0.0):
    pos = np.asarray(ans["positions"], dtype=np.float64)
    t = pos.shape[0]
    rep = lambda a: np.repeat(np.asarray(a, dtype=np.float64)[None], t, axis=0)
    return states_payload(ans["scene_id"], pos + jitter, rep(ans["scale"]), rep(ans["opacity"]), rep(ans["color"]))


def build_bruteforce(dst):
    """Enumerate public-generator seeds and copy the best-matching scene's true states."""
    gen = _load_scenes_module()
    log = {}
    for exp in EXPERIMENTS:
        cands = list(range(500)) + list(gen.TRAIN_SEEDS[exp])
        frames = {}
        for seed in cands:
            inputs, answers = gen.make_scene(seed, exp)
            frames[seed] = (np.asarray(inputs["frames"]), answers)
        for path in scene_set("test", exp):
            obs = np.asarray(json.loads(path.read_text())["frames"])
            errs = {s: float(np.mean((f - obs) ** 2)) for s, (f, _) in frames.items()}
            best = min(errs, key=errs.get)
            log[path.stem] = {"best_seed": best, "frame_mse": errs[best], "median_mse": float(np.median(list(errs.values())))}
            ans = dict(frames[best][1], scene_id=path.stem)
            write_pred(dst, exp, path.stem, answers_payload(ans))
    return log


def build_init_points(dst):
    for exp in EXPERIMENTS:
        for path in scene_set("test", exp):
            raw = json.loads(path.read_text())
            t = int(raw["num_frames"])
            pos = np.asarray(raw["init_points"]["positions"])
            col = np.asarray(raw["init_points"]["colors"])
            k = pos.shape[0]
            write_pred(
                dst,
                exp,
                path.stem,
                states_payload(path.stem, np.repeat(pos[None], t, 0), np.full((t, k), 0.04), np.full((t, k), 0.6), np.repeat(col[None], t, 0)),
            )


def build_oracle(dst, jitter=0.0):
    for exp in EXPERIMENTS:
        for path in sorted((ROOT / "eval" / "test" / "scenes" / exp).glob("*.json")):
            write_pred(dst, exp, path.stem, answers_payload(json.loads(path.read_text()), jitter))


def _set_states(payload, **arrays):
    for k, v in arrays.items():
        payload["frozen_states"][k] = v
    return payload


def case_specs():
    """case name -> (kind, {dir_name: source}) ; source is a path or a builder callable."""
    m = METHODS
    specs = {
        # scientific cases
        "reference": ("science", {"splannequin": REF_DIR}),
        "reference_seed1": ("science", {"splannequin": m / "splannequin_seed1"}),
        "reference_seed2": ("science", {"splannequin": m / "splannequin_seed2"}),
        "baseline_4dgaussians": ("science", {"splannequin": m / "baseline"}),
        "weaker_no_hidden": ("science", {"splannequin": m / "no_hidden"}),
        "weaker_no_defective": ("science", {"splannequin": m / "no_defective"}),
        "better_tau20": ("science", {"splannequin": m / "splannequin_tau20"}),
        "sensitivity_lambda1": ("science", {"splannequin": m / "splannequin_lambda1"}),
        "better_nearest_anchor_snap": ("science", {"splannequin": m / "splannequin_snap"}),
        "baseline_nearest_anchor_snap": ("science", {"splannequin": m / "baseline_snap"}),
        "shortcut_static_reconstruction": ("science", {"splannequin": m / "static"}),
        "simple_init_points": ("science", {"splannequin": "init_points"}),
        # shortcut / robustness cases
        "shortcut_bruteforce_public_seeds": ("shortcut", {"splannequin": "bruteforce"}),
        "shortcut_forged_fields": ("shortcut", {"splannequin": ("modify", "forged")}),
        "shortcut_black_output": ("shortcut", {"splannequin": ("modify", "black")}),
        "shortcut_null_predictions": ("shortcut", {"splannequin": ("modify", "null")}),
        "shortcut_infinite_predictions": ("shortcut", {"splannequin": ("modify", "inf")}),
        "shortcut_empty_predictions": ("shortcut", {"splannequin": ("modify", "empty")}),
        "shortcut_over_budget": ("shortcut", {"splannequin": ("modify", "double")}),
        "shortcut_out_of_range_opacity": ("shortcut", {"splannequin": ("modify", "opacity")}),
        "shortcut_missing_scene": ("shortcut", {"splannequin": ("modify", "drop_scene")}),
        "shortcut_omit_proposed": ("shortcut", {"baseline": m / "baseline"}),
        "shortcut_failed_method_baseline_only": ("shortcut", {"baseline": REF_DIR, "splannequin": ("modify", "black")}),
        "shortcut_extra_method": ("shortcut", {"splannequin": REF_DIR, "junk_method": ("modify", "black")}),
        "shortcut_stale_scores": ("shortcut", {"splannequin": REF_DIR}),
        "audit_oracle_answers": ("audit", {"splannequin": "oracle"}),
        "audit_oracle_plus_1e-4": ("audit", {"splannequin": "oracle_jitter"}),
    }
    return specs


def _modify(kind):
    def fn(p):
        fs = p["frozen_states"]
        pos = np.asarray(fs["positions"])
        if kind == "forged":
            p.update({"psnr": 1e9, "success": True, "inference_state_values": -1e9, "metrics": {"psnr": 100, "ssim": 1}})
        elif kind == "black":
            fs["opacity"] = np.zeros_like(np.asarray(fs["opacity"])).tolist()
        elif kind == "null":
            fs["positions"] = [[[None, None] for _ in row] for row in fs["positions"]]
        elif kind == "inf":
            fs["positions"] = np.full_like(pos, np.inf).tolist()
        elif kind == "empty":
            t = pos.shape[0]
            _set_states(p, positions=[[] for _ in range(t)], scale=[[] for _ in range(t)], opacity=[[] for _ in range(t)], color=[[] for _ in range(t)])
        elif kind == "double":
            for k in ("positions", "scale", "opacity", "color"):
                a = np.asarray(fs[k])
                fs[k] = np.concatenate([a, a], axis=1).tolist()
        elif kind == "opacity":
            fs["opacity"] = np.minimum(1.5, np.asarray(fs["opacity"]) * 3.0).tolist()
        return p

    return fn


def build_case(name, entries):
    case_dir = CASES / name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    ckpt = case_dir / "ckpt"
    ckpt.mkdir(parents=True)
    notes = {}
    for dir_name, src in entries.items():
        dst = ckpt / dir_name
        if isinstance(src, Path):
            os.symlink(src, dst)
        elif src == "bruteforce":
            notes["bruteforce"] = build_bruteforce(dst)
        elif src == "init_points":
            build_init_points(dst)
        elif src == "oracle":
            build_oracle(dst)
        elif src == "oracle_jitter":
            build_oracle(dst, jitter=1e-4)
        elif isinstance(src, tuple) and src[0] == "modify":
            kind = src[1]
            if kind == "drop_scene":
                first = (EXPERIMENTS[0], test_scene_ids(EXPERIMENTS[0])[0])
                copy_modified(REF_DIR, dst, lambda p: p, drop={first})
            else:
                copy_modified(REF_DIR, dst, _modify(kind))
    return case_dir, ckpt, notes


def score_case(name, entries):
    case_dir, ckpt, notes = build_case(name, entries)
    out = case_dir / "scores_test.json"
    if name == "shortcut_stale_scores":
        stale = {"experiments": {e: {"results": {SLOT: {"psnr": 100.0, "ssim": 1.0}}} for e in EXPERIMENTS}}
        out.write_text(json.dumps(stale))
    start = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "test" / "evaluate.py"), "--checkpoints", str(ckpt), "--out", str(out), "--details", str(case_dir / "details_test.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    record = {
        "case": name,
        "hidden_returncode": proc.returncode,
        "hidden_seconds": round(time.time() - start, 2),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "notes": notes,
    }
    if out.exists():
        record["scores_test"] = json.loads(out.read_text())
    # Train-side scores for cases whose method directory also holds training-scene outputs.
    src = entries.get("splannequin")
    if isinstance(src, Path) and (src / "predictions").exists() and any((src / "predictions" / EXPERIMENTS[0]).glob("*_10[1-4].json")):
        tout = case_dir / "scores_train.json"
        tproc = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "train" / "evaluate.py"), "--checkpoints", str(ckpt), "--out", str(tout), "--details", str(case_dir / "details_train.json")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        record["train_returncode"] = tproc.returncode
        if tout.exists():
            record["scores_train"] = json.loads(tout.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"[cases] {name}: rc={proc.returncode}", flush=True)
    return record


def cmd_cases():
    CASES.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, (_, entries) in case_specs().items():
        needed = [s for s in entries.values() if isinstance(s, Path)]
        if any(not (s / "predictions").exists() for s in needed):
            missing.append(name)
            print(f"[cases] {name}: skipped (training output missing)", flush=True)
            continue
        score_case(name, entries)
    return 0


# ----------------------------------------------------------------------------- summary


def _metric_table(record):
    sc = record.get("scores_test", {}).get("experiments", {})
    return {exp: sc.get(exp, {}).get("results", {}).get(SLOT) for exp in EXPERIMENTS}


def cmd_summary():
    ref_cfg = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    records = {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))}
    specs = case_specs()
    tables = {name: _metric_table(r) for name, r in records.items()}
    ref = tables.get("reference")
    if not ref or not all(ref.values()):
        print("[summary] reference results missing")
        return 1
    science = [n for n in records if specs.get(n, ("",))[0] == "science" and n != "simple_init_points" and all(tables[n].values())]
    sd = {}
    for exp in EXPERIMENTS:
        for metric in ref_cfg[exp]["metrics"]:
            vals = [tables[n][exp][metric] for n in science]
            sd[(exp, metric)] = float(np.std(vals)) if len(vals) > 1 else 0.0

    def delta(name):
        tab = tables[name]
        if not all(tab.values()):
            return None
        tot = 0.0
        for exp in EXPERIMENTS:
            w = ref_cfg[exp]["weight"]
            for metric, spec in ref_cfg[exp]["metrics"].items():
                s = sd[(exp, metric)]
                if s <= 0:
                    continue
                sign = 1.0 if spec["higher_is_better"] else -1.0
                tot += w * abs(spec["coefficient"]) * sign * (tab[exp][metric] - ref[exp][metric]) / s
        return tot

    def paired(name):
        out = {}
        try:
            det_ref = json.loads((CASES / "reference" / "details_test.json").read_text())[SLOT]
            det = json.loads((CASES / name / "details_test.json").read_text())[SLOT]
        except (OSError, KeyError, ValueError):
            return None
        if not det or not det_ref:
            return None
        for exp in EXPERIMENTS:
            try:
                a = np.concatenate([d["per_freeze_psnr"] for d in det if d["experiment"] == exp])
                b = np.concatenate([d["per_freeze_psnr"] for d in det_ref if d["experiment"] == exp])
            except ValueError:
                return None
            if a.shape != b.shape:
                return None
            diff = a - b
            out[exp] = {
                "n_videos": int(diff.size),
                "mean_psnr_diff": float(diff.mean()),
                "se": float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0,
                "frac_better": float(np.mean(diff > 0)),
            }
        return out

    summary = {"normalization_sd": {f"{e}.{m}": v for (e, m), v in sd.items()}, "cases": {}}
    for name, rec in records.items():
        summary["cases"][name] = {
            "kind": specs.get(name, ("?",))[0],
            "hidden_returncode": rec["hidden_returncode"],
            "failed_methods": rec.get("scores_test", {}).get("metadata", {}).get("failed_methods"),
            "test": tables[name],
            "train": (rec.get("scores_train", {}).get("experiments", {}) or None)
            and {e: rec["scores_train"]["experiments"][e]["results"].get(SLOT) for e in EXPERIMENTS},
            "illustrative_rank_delta": delta(name),
            "paired_psnr_vs_reference": paired(name) if name != "reference" else None,
            "notes": rec.get("notes"),
        }
    seeds = [n for n in ("reference", "reference_seed1", "reference_seed2") if n in tables and all(tables[n].values())]
    spread = {}
    for exp in EXPERIMENTS:
        for metric in ref_cfg[exp]["metrics"]:
            vals = [tables[n][exp][metric] for n in seeds]
            spread[f"{exp}.{metric}"] = {"values": vals, "range": float(max(vals) - min(vals)) if vals else None}
    deltas = [delta(n) for n in seeds]
    spread["illustrative_rank_delta"] = {"values": deltas, "range": float(max(deltas) - min(deltas)) if deltas else None}
    summary["seed_spread"] = spread
    (ROOT / "redteam" / "hardening" / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))

    lines = ["| case | kind | rc | " + " | ".join(f"{e[:14]} {m}" for e in EXPERIMENTS for m in ("psnr", "ssim", "motion_preservation_error")) + " | rank delta |", "|" + "---|" * 10]
    for name, c in summary["cases"].items():
        vals = []
        for e in EXPERIMENTS:
            t = c["test"].get(e) or {}
            vals += [f"{t.get(m, float('nan')):.4f}" if t else "-" for m in ("psnr", "ssim", "motion_preservation_error")]
        d = c["illustrative_rank_delta"]
        lines.append(f"| {name} | {c['kind']} | {c['hidden_returncode']} | " + " | ".join(vals) + f" | {d:+.3f} |" if d is not None else f"| {name} | {c['kind']} | {c['hidden_returncode']} | " + " | ".join(vals) + " | n/a |")
    (ROOT / "redteam" / "hardening" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(spread, indent=1))
    for name, c in summary["cases"].items():
        if c["paired_psnr_vs_reference"]:
            print(name, json.dumps(c["paired_psnr_vs_reference"]))
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    rc = {"train": cmd_train, "cases": cmd_cases, "summary": cmd_summary}[cmd]()
    sys.exit(rc)


if __name__ == "__main__":
    main()
