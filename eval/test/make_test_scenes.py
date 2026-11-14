"""Build the hidden test scenes (run once by the gym maintainer).

Uses private seeds that live only here (40 near-static and 16 large-motion
scenes; regenerated in hardening iteration 12 for the crowded scene layout and
the larger near-static test). Writes:
  * eval/test/scenes/<experiment>/<scene_id>.json  -- answers (true per-frame
    scene states, camera poses, private freeze instants); never visible.
  * data/test/<experiment>/<scene_id>.json         -- method inputs only
    (noisy monocular frames, camera poses, timestamps, sparse init points).

The generator code itself is the public one in data/scenes.py; without these
seeds it cannot regenerate the test answers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PRIVATE_SEEDS = {
    "synthetic_validation_paper": [
        898001217869023804,
        3656137533311011161,
        948944453430394740,
        67644973833449989,
        2848386546188400019,
        3450398819006990654,
        2572487468356201899,
        177097578316152889,
        1141017321357365740,
        1744986429373814438,
        3013677836522461884,
        2851021632363979885,
        4174035352535907073,
        2910172397818349864,
        4410182964081535536,
        2523460861468486907,
        402256494481067622,
        607286029366412167,
        4597189365716468231,
        4298021352483320702,
        1876234613405539670,
        1292180584884191094,
        1016532972557999381,
        2631456199442686843,
        2973570403472308587,
        470165316591905519,
        3102134806843380612,
        2429376819073475573,
        235147189701080237,
        653087791535847856,
        3963308945253638101,
        2137238987929631683,
        103890951492017640,
        2644581165077455739,
        3562031918316500470,
        2568135606773921394,
        2365265115926939642,
        1814988474083646761,
        3589936272732814130,
        1769260122947475176,
    ],
    "synthetic_large_motion": [
        2239696490441333478,
        2325204565705472527,
        2275341357877672862,
        2339080992054502921,
        4367019637125499351,
        2866129434359779362,
        2279351753160771014,
        4066417311551910244,
        2376941156926851788,
        3486063902295575705,
        4412093121430281276,
        1299921945563442460,
        3426773499952564609,
        3918898056294428620,
        1892403334900017954,
        1378494091458623381,
    ],
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("bench_scenes", ROOT / "data" / "scenes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-root", default=str(ROOT / "eval" / "test" / "scenes"))
    parser.add_argument("--inputs-root", default=str(ROOT / "data" / "test"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gen = _load_generator()
    for experiment, seeds in PRIVATE_SEEDS.items():
        ans_dir = Path(args.answers_root) / experiment
        inp_dir = Path(args.inputs_root) / experiment
        ans_dir.mkdir(parents=True, exist_ok=True)
        inp_dir.mkdir(parents=True, exist_ok=True)
        for idx, seed in enumerate(seeds):
            scene_id = f"{experiment}_test_{idx:02d}"
            ans_path = ans_dir / f"{scene_id}.json"
            inp_path = inp_dir / f"{scene_id}.json"
            if ans_path.exists() and inp_path.exists() and not args.force:
                continue
            inputs, answers = gen.make_scene(seed, experiment)
            inputs["scene_id"] = scene_id
            answers["scene_id"] = scene_id
            inp_path.write_text(json.dumps(inputs), encoding="utf-8")
            ans_path.write_text(json.dumps(answers), encoding="utf-8")
            print(f"wrote {inp_path} and answers")


if __name__ == "__main__":
    main()
