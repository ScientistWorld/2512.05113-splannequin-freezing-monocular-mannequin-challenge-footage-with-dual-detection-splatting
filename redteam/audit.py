"""Isolated audit driver; evaluator is always a fresh, unmodified subprocess."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
BASE = ROOT / 'redteam'
ORIGINAL = BASE / 'original'
EXPERIMENT = 'synthetic_validation_paper'
CASES = [
    'reference_existing',
    'reference_seed_7', 'reference_seed_17', 'reference_seed_27',
    'better_initialization', 'better_more_training', 'better_longer_small_lr',
    'weaker_short_training', 'weaker_remove_anchors',
    'shortcut_copy_gt', 'shortcut_copy_input', 'shortcut_rebuild_seed',
    'shortcut_psnr_epsilon',
    'shortcut_bruteforce_seed', 'shortcut_static_reconstruction',
    'shortcut_zero_state', 'shortcut_negative_state', 'shortcut_baseline_inflation',
    'shortcut_forged_metrics', 'shortcut_replace_fixed_target',
    'shortcut_zero_target_and_outputs', 'shortcut_constant_zero',
    'shortcut_null_predictions', 'shortcut_infinite_predictions',
    'shortcut_empty_predictions', 'shortcut_omit_proposed',
    'shortcut_failed_method_baseline_only',
    'shortcut_extra_method_ignored', 'shortcut_extra_scenes_ignored',
    'shortcut_failed_stale_scores',
]
RESTORE_FILES = [
    'data/synthetic_mannequin.json', 'data/extra_seed_11.json', 'data/extra_seed_29.json',
    'method/splannequin_1d.py', 'scoring/scores.json',
    'scoring/scores_train.json', 'scoring/scores_test.json',
]


def restore():
    for name in RESTORE_FILES:
        shutil.copyfile(ORIGINAL / name, ROOT / name)
    checkpoint_dir = ROOT / 'checkpoints'
    shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()
    for source in (ORIGINAL / 'artifacts').rglob('render_output.json'):
        destination = checkpoint_dir / source.relative_to(ORIGINAL / 'artifacts').parent / 'predictions.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (ROOT / 'method/audit_submission.py').unlink(missing_ok=True)


def evaluator_hashes():
    paths = sorted((ROOT / 'eval').rglob('*.py')) + [ROOT / 'scripts/evaluate_test.sh']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def invoke(command, path):
    with path.open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def run_case(case):
    out = BASE / case
    out.mkdir(parents=True, exist_ok=True)
    restore()
    hashes = evaluator_hashes()
    shutil.copyfile(BASE / 'payload.py', ROOT / 'method/audit_submission.py')
    start = time.monotonic()
    preparation_rc = invoke([sys.executable, 'method/audit_submission.py', '--case', case,
                             '--stage', 'prepare'], out / 'training.log')
    if preparation_rc:
        raise RuntimeError(f'{case}: preparation failed; inspect training.log')
    train_rc = invoke(['bash', 'scripts/evaluate_train.sh'], out / 'train_evaluator.log')
    # Even invalid submissions must be exercised through the hidden test.
    if train_rc == 0:
        subprocess.run([sys.executable, 'method/audit_submission.py', '--case', case,
                        '--stage', 'finalize'], cwd=ROOT, check=True)
    if case != 'shortcut_failed_stale_scores':
        (ROOT / 'scoring/scores_test.json').unlink(missing_ok=True)
    proposed = json.loads((ROOT / 'checkpoints/splannequin/predictions.json').read_text())
    selected_keys = ['variant', 'steps', 'num_primitives', 'inference_state_values',
                     'training_summary', 'shortcut_search']
    descriptor = {key: proposed[key] for key in selected_keys if key in proposed}
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'checkpoints').rglob('predictions.json'))
    hidden_rc = invoke(['bash', 'scripts/evaluate_test.sh'], out / 'evaluator.log')
    written = (ROOT / 'scoring/scores_test.json').exists()
    for filename in ['scores_train.json', 'scores_test.json']:
        source = ROOT / 'scoring' / filename
        if source.exists():
            shutil.copyfile(source, out / filename)
        else:
            (out / filename).unlink(missing_ok=True)
    metadata = {
        'case': case, 'preparation_rc': preparation_rc, 'train_rc': train_rc,
        'hidden_rc': hidden_rc, 'score_file_exists': written,
        'score_file_written_by_hidden_test': written and hidden_rc == 0,
        'elapsed_seconds': time.monotonic() - start,
        'prediction_files_on_disk': files, 'proposed_artifact': descriptor,
        'evaluator_sha256_before': hashes,
        'evaluator_sha256_after': evaluator_hashes(),
        'hidden_command': ['bash', 'scripts/evaluate_test.sh'],
        'fresh_process': True,
    }
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    # Preserve compact diagnostics even for ignored logs and failed evaluation.
    for filename in ['evaluator.log', 'train_evaluator.log']:
        content = (out / filename).read_text()
        if content:
            (out / filename.replace('.log', '_output.txt')).write_text(content[-6000:])
    print(f'{case}: hidden_rc={hidden_rc}, seconds={metadata["elapsed_seconds"]:.1f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.restore:
        restore()
        return
    try:
        for case in CASES if args.all else [args.case]:
            if args.all and (BASE / case / 'run_metadata.json').exists():
                print(f'{case}: already completed', flush=True)
                continue
            run_case(case)
    finally:
        restore()


if __name__ == '__main__':
    main()
