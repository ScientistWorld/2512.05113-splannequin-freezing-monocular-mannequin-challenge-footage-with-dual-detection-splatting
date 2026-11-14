"""Pilot for hardening iteration 12 (design check before the full audit).

Checks, on 8 freshly generated public scenes per regime of the revised scene
layout (a group of people across the walking line near the start), whether
  * Splannequin still beats its 4DGaussians baseline,
  * removing the defective (future) anchors or the hidden (past) anchors costs
    quality, in the paper's direction (Table 3),
  * the defective gradient test at the paper's literal 1e-9 on this image line
    or at 1e-9 expressed for the paper's 640x360 RGB frames is needed,
  * candidate "better" variants are still better, static stays below,
and measures the wall time of a 30k-iteration batch.

  python redteam/hardening_v2/pilot.py run       # generate scenes, train every variant (resumable)
  python redteam/hardening_v2/pilot.py summary   # score with the train evaluator's functions
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PW = ROOT / "tmp" / "pilot"
SCENES = PW / "scenes"
CKPT = PW / "ckpt"
LOGS = PW / "logs"
DONE = PW / "done"
OUT = ROOT / "redteam" / "hardening_v2" / "pilot_results.json"
PAPER, LARGE = "synthetic_validation_paper", "synthetic_large_motion"
SEEDS = {PAPER: list(range(301, 309)), LARGE: list(range(401, 409))}
SCALED = ["--grad-loss-values", "691200"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gen_scenes():
    gen = load(ROOT / "data" / "scenes.py", "bench_scenes")
    for exp, seeds in SEEDS.items():
        d = SCENES / exp
        d.mkdir(parents=True, exist_ok=True)
        for i, seed in enumerate(seeds):
            p = d / f"scene_{i:02d}.json"
            if p.exists():
                continue
            inputs, answers = gen.make_scene(seed, exp)
            inputs["answers"] = answers
            p.write_text(json.dumps(inputs), encoding="utf-8")


TASKS = [
    # name, variant, seed, extra args, experiment
    ("ref_scaled", "splannequin", 0, SCALED, PAPER),
    ("nodef_scaled", "no_defective", 0, SCALED, PAPER),
    ("nohid_scaled", "no_hidden", 0, SCALED, PAPER),
    ("ref_literal", "splannequin", 0, [], PAPER),
    ("nodef_literal", "no_defective", 0, [], PAPER),
    ("baseline", "baseline", 0, [], PAPER),
    ("ref_scaled_seed1", "splannequin", 1, SCALED, PAPER),
    ("lambda1_scaled", "splannequin", 0, SCALED + ["--lambda-hidden", "1", "--lambda-defective", "1"], PAPER),
    ("smooth10_scaled", "splannequin", 0, SCALED + ["--time-smoothness-weight", "0.1"], PAPER),
    ("static", "static", 0, [], PAPER),
    ("large_ref_scaled", "splannequin", 0, SCALED, LARGE),
    ("large_nodef_scaled", "no_defective", 0, SCALED, LARGE),
    ("large_baseline", "baseline", 0, [], LARGE),
    ("large_static", "static", 0, [], LARGE),
]


def run_task(task):
    name, variant, seed, extra, exp = task
    marker = DONE / f"{name}.json"
    if marker.exists():
        return name, 0
    scenes = [str(p) for p in sorted((SCENES / exp).glob("*.json"))]
    cmd = [sys.executable, str(ROOT / "method" / "splannequin.py"), "--variant", variant, "--seed", str(seed),
           "--out-dir", str(CKPT / name), *extra, "--scenes", *scenes]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.time()
    with open(LOGS / f"{name}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    secs = time.time() - start
    if rc == 0:
        marker.write_text(json.dumps({"seconds": round(secs, 1), "cmd": cmd[1:]}))
    print(f"[pilot] {name}: rc={rc} {secs:.0f}s", flush=True)
    return name, rc


def cmd_run():
    for d in (CKPT, LOGS, DONE):
        d.mkdir(parents=True, exist_ok=True)
    gen_scenes()
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    print(f"[pilot] {len(TASKS)} tasks, cpus={ncpu}", flush=True)
    with ThreadPoolExecutor(max_workers=len(TASKS)) as ex:
        res = list(ex.map(run_task, TASKS))
    return 1 if any(rc for _, rc in res) else 0


def cmd_summary():
    ev = load(ROOT / "eval" / "train" / "evaluate.py", "train_eval")
    scenes = ev.load_answer_scenes(SCENES)
    out = {"tasks": {}}
    for name, variant, seed, extra, exp in TASKS:
        d = CKPT / name
        if not (d / "predictions").exists():
            continue
        det = []
        try:
            means = ev.evaluate_experiment(d, exp, scenes[exp], det)
        except Exception as exc:  # noqa: BLE001
            out["tasks"][name] = {"error": str(exc)}
            continue
        summ = [json.loads(p.read_text())["training_summary"] for p in sorted((d / "predictions" / exp).glob("*.json"))][0]
        done = json.loads((DONE / f"{name}.json").read_text()) if (DONE / f"{name}.json").exists() else {}
        out["tasks"][name] = {
            "experiment": exp,
            "means": means,
            "per_scene_psnr": [x["psnr"] for x in det],
            "per_freeze_psnr": [x["per_freeze_psnr"] for x in det],
            "per_scene_selectivity": [x["instant_selectivity"] for x in det],
            "training_summary": summ,
            "wall_seconds": done.get("seconds"),
        }

    def paired(a, b):
        if a not in out["tasks"] or b not in out["tasks"]:
            return None
        x = np.asarray(out["tasks"][a]["per_scene_psnr"]) - np.asarray(out["tasks"][b]["per_scene_psnr"])
        fa = np.asarray(out["tasks"][a]["per_freeze_psnr"]) - np.asarray(out["tasks"][b]["per_freeze_psnr"])
        k = fa.shape[1]
        return {
            "mean": float(x.mean()),
            "se": float(x.std(ddof=1) / np.sqrt(x.size)),
            "early_third": float(fa[:, : k // 3].mean()),
            "middle_third": float(fa[:, k // 3 : 2 * k // 3].mean()),
            "late_third": float(fa[:, 2 * k // 3 :].mean()),
            "scenes_better": int((x > 0).sum()),
        }

    comps = {
        "ref_scaled - baseline": ("ref_scaled", "baseline"),
        "ref_scaled - nodef_scaled": ("ref_scaled", "nodef_scaled"),
        "ref_scaled - nohid_scaled": ("ref_scaled", "nohid_scaled"),
        "ref_literal - baseline": ("ref_literal", "baseline"),
        "ref_literal - nodef_literal": ("ref_literal", "nodef_literal"),
        "ref_scaled - ref_literal": ("ref_scaled", "ref_literal"),
        "ref_scaled_seed1 - ref_scaled": ("ref_scaled_seed1", "ref_scaled"),
        "lambda1_scaled - ref_scaled": ("lambda1_scaled", "ref_scaled"),
        "smooth10_scaled - ref_scaled": ("smooth10_scaled", "ref_scaled"),
        "static - ref_scaled": ("static", "ref_scaled"),
        "large: ref_scaled - baseline": ("large_ref_scaled", "large_baseline"),
        "large: ref_scaled - nodef_scaled": ("large_ref_scaled", "large_nodef_scaled"),
        "large: static - ref_scaled": ("large_static", "large_ref_scaled"),
    }
    out["paired_psnr"] = {k: paired(*v) for k, v in comps.items()}
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    for name, t in out["tasks"].items():
        if "means" in t:
            m = t["means"]
            s = t["training_summary"]
            print(f"{name:20s} psnr {m['psnr']:.3f} ssim {m['ssim']:.4f} lpips {m['lpips']:.4f} fvd {m['fvd']:.2f} "
                  f"edge {m['edge_mae']:.4f} sel {m['instant_selectivity']:.4f} | hid {s.get('hidden_anchor_items')} "
                  f"def {s.get('defective_anchor_items')} defseen {s.get('defective_states_seen')} wall {t['wall_seconds']}")
    for k, v in out["paired_psnr"].items():
        print(k, json.dumps(v))
    return 0


if __name__ == "__main__":
    sys.exit({"run": cmd_run, "summary": cmd_summary}[sys.argv[1] if len(sys.argv) > 1 else "summary"]())
