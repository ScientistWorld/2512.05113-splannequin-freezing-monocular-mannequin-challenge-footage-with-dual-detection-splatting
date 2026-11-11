"""Lightweight evidence checks; no model computation or evaluator changes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'redteam'


def read(path):
    return json.loads(path.read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    start = (BASE / 'original/start_commit.txt').read_text().strip()
    summary = read(BASE / 'summary.json')
    visibility = read(BASE / 'visibility_result.json')
    assert visibility == {'shortcut_cases_rerun': 21, 'changed_score_files': [],
                          'all_shortcuts_explicitly_create_visible_method_list': True}
    metadata_paths = sorted(BASE.glob('*/run_metadata.json'))
    assert len(metadata_paths) == 30
    assert len([p for p in metadata_paths if p.parent.name.startswith('shortcut_')]) == 21
    failures, nonfinite, runs = [], [], {}
    for path in metadata_paths:
        name = path.parent.name
        metadata = read(path)
        assert (path.parent / 'run.sh').is_file(), name
        assert metadata['preparation_rc'] == 0, name
        assert metadata['hidden_command'] == ['bash', 'scripts/evaluate_test.sh'], name
        assert metadata['fresh_process'], name
        if name.startswith('shortcut_'):
            assert metadata['submission_writes_minimal_method_source'], name
            source = metadata['method_source_at_test']['experiments']['synthetic_validation_paper']['results']
            assert source == {'4dgaussians': {}, '4dgaussians_splannequin': {}}, name
        assert metadata['evaluator_sha256_before'] == metadata['evaluator_sha256_after'], name
        for relative, expected in metadata['evaluator_sha256_before'].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
            assert hashlib.sha256(git('show', f'{start}:{relative}')).hexdigest() == expected, relative
        if metadata['hidden_rc']:
            failures.append(name)
            assert not metadata['score_file_written_by_hidden_test'], name
        else:
            assert metadata['score_file_written_by_hidden_test'], name
            values = read(path.parent / 'scores_test.json')['experiments']['synthetic_validation_paper']['results']
            if any(not math.isfinite(v) for metrics in values.values() for v in metrics.values()):
                nonfinite.append(name)
        runs[name] = {
            'hidden_exit_code': metadata['hidden_rc'],
            'scores_written_by_hidden_test': metadata['score_file_written_by_hidden_test'],
            'evaluator_unchanged': True,
            'seconds': metadata['elapsed_seconds'],
        }
    assert set(failures) == {'shortcut_infinite_predictions', 'shortcut_empty_predictions',
                             'shortcut_failed_stale_scores'}
    assert nonfinite == ['shortcut_null_predictions']
    scenes = {}
    for name in ['synthetic_mannequin.json', 'extra_seed_11.json', 'extra_seed_29.json']:
        scene = read(ROOT / 'data' / name)
        observations_equal_target = scene['frames'] == scene['gt_freeze']
        states = scene['true_states']
        static = all(state == states[0] for state in states)
        assert observations_equal_target and static, name
        scenes[name] = {'observations_equal_freeze_target': observations_equal_target,
                        'all_true_states_equal': static,
                        'frame_count': len(scene['frames']), 'width': scene['width']}
    excluded = ['action.yaml', 'README.md', 'manifest.json']
    tracked = git('ls-tree', '-r', '--name-only', start).decode().splitlines()
    gym_files = [p for p in tracked if p not in excluded and not p.startswith('trajectory/')]
    changed = git('diff', '--name-only', start, '--', *gym_files).decode().splitlines()
    assert not changed, changed
    index_paths = git('ls-files').decode().splitlines()
    unexpected_additions = [p for p in index_paths if p not in tracked
                            and not p.startswith('redteam/') and p != 'REDTEAM.md']
    assert not unexpected_additions, unexpected_additions
    exact = summary['runs']['shortcut_copy_gt']['results']['4dgaussians_splannequin']
    epsilon = summary['runs']['shortcut_psnr_epsilon']['results']['4dgaussians_splannequin']
    assert epsilon['psnr'] > exact['psnr'] and epsilon['lpips'] > exact['lpips']
    for name in ['shortcut_omit_proposed', 'shortcut_failed_method_baseline_only']:
        assert summary['runs'][name]['missing_reference_methods'] == ['4dgaussians_splannequin']
    output = {
        'starting_commit': start,
        'completed_cases': len(metadata_paths), 'shortcut_cases': 21,
        'total_hidden_test_invocations_including_visibility_rerun': 51,
        'visibility_rerun': visibility,
        'successful_hidden_test_exits': len(metadata_paths) - len(failures),
        'failed_hidden_test_cases': failures, 'nonfinite_success_cases': nonfinite,
        'all_case_evaluators_match_starting_git_files': True,
        'all_hidden_commands_have_no_arguments_and_run_in_fresh_processes': True,
        'tracked_gym_files_changed_from_start': changed,
        'unexpected_tracked_additions_outside_audit': unexpected_additions,
        'restoration_exclusions': {'protocol': ['action.yaml'],
                                   'preexisting_system_artifacts': ['README.md', 'manifest.json', 'trajectory/']},
        'scenes': scenes, 'runs': runs,
    }
    (BASE / 'integrity_checks.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f'Verified {len(metadata_paths)} cases, 21 shortcuts, unchanged evaluators, and restored gym files.')


if __name__ == '__main__':
    main()
