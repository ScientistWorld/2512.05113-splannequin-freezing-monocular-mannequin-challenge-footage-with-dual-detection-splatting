"""Build the hidden test scenes (run once by the gym maintainer).

Uses private seeds that live only here. Writes:
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
        2239578907864309142,
        863104004164453038,
        1747622069643925534,
        504801963601108653,
        4132320482572989405,
        630701741547371949,
        235782449501310223,
        2325571966136567060,
        509830130551314787,
        1057761822664096671,
        1302157476834357641,
        3702920797835461187,
        4439252778400456878,
        670768381788072349,
        3495627358930762288,
        3499215611909091358,
    ],
    "synthetic_large_motion": [
        4279955927991666583,
        3768918705289910822,
        829694960040219271,
        2418741188145251216,
        1518392862033418015,
        3678850898514446969,
        3224905530709213926,
        1167965616827067096,
        1327378123768485842,
        1587523512549474382,
        571000192363327468,
        3684844305154521078,
        1802783646629704883,
        827920258654718321,
        1336115092411817884,
        2949218111402476609,
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
