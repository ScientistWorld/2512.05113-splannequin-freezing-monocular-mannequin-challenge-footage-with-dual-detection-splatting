"""Summarize actual hidden-test artifacts, never submission-reported metrics.

The actual future-attempt variance is unknowable at audit time. We supply
raw vectors, Pareto comparisons, and explicitly labelled cohort-normalized
illustrations using the user-specified weights and metric directions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

BASE = Path('/home/user/redteam')
EXP = 'synthetic_validation_paper'
PROPOSED = '4dgaussians_splannequin'


def load(path):
    return json.loads(path.read_text())


def results(scores):
    return scores['experiments'][EXP]['results']


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def main():
    ref = load(Path('/home/user/scoring/reference.json'))
    config = ref['experiments'][EXP]
    metric_config = config['metrics']
    runs = {}
    for path in sorted(BASE.glob('*/run_metadata.json')):
        name = path.parent.name
        metadata = load(path)
        scores_path = path.parent / 'scores_test.json'
        scores = load(scores_path) if scores_path.exists() else None
        runs[name] = {'metadata': metadata, 'scores': scores}
    reference = results(runs['reference_existing']['scores'])
    genuine = [name for name in runs if not name.startswith('shortcut_')]

    def scales(names):
        out = {}
        for method, metrics in reference.items():
            for metric in metrics:
                values = []
                for name in names:
                    run = runs[name]
                    if run['metadata']['hidden_rc'] != 0 or run['scores'] is None:
                        continue
                    v = results(run['scores']).get(method, {}).get(metric)
                    if finite(v):
                        values.append(v)
                sd = statistics.pstdev(values) if values else 0.0
                out[f'{method}/{metric}'] = sd if sd else None
        return out

    quality_scales = scales(genuine)
    all_scales = scales(list(runs))

    def delta(scores, metric_scales):
        total = 0.0
        terms = {}
        for method, metrics in results(scores).items():
            for metric, value in metrics.items():
                baseline = reference.get(method, {}).get(metric)
                sd = metric_scales.get(f'{method}/{metric}')
                if baseline is None or not finite(value):
                    continue
                d = (value - baseline) * (1 if metric_config[metric]['higher_is_better'] else -1)
                if not sd:
                    if d:
                        raise ValueError(f'Nonzero change with no variance: {method}/{metric}')
                    continue
                term = config['weight'] * abs(metric_config[metric]['coefficient']) * d / sd
                terms[f'{method}/{metric}'] = term
                total += term
        return total, terms

    table = {}
    for name, run in runs.items():
        metadata, scores = run['metadata'], run['scores']
        row = {'hidden_rc': metadata['hidden_rc'], 'train_rc': metadata['train_rc'],
               'score_file_written_by_hidden_test': metadata['score_file_written_by_hidden_test']}
        if scores and metadata['hidden_rc'] == 0:
            current = results(scores)
            row['results'] = current
            row['nonfinite_metrics'] = [f'{method}/{metric}' for method, metrics in current.items()
                                        for metric, value in metrics.items() if not finite(value)]
            if not row['nonfinite_metrics']:
                row['audit_cohort_normalized_delta'], row['audit_cohort_terms'] = delta(scores, all_scales)
                if name in genuine:
                    row['scientific_cohort_normalized_delta'], _ = delta(scores, quality_scales)
            row['missing_reference_methods'] = sorted(set(reference) - set(current))
            wins = losses = 0
            for method, metrics in reference.items():
                for metric, baseline in metrics.items():
                    value = current.get(method, {}).get(metric)
                    if not finite(value):
                        continue
                    d = (value-baseline) * (1 if metric_config[metric]['higher_is_better'] else -1)
                    wins += d > 1e-12
                    losses += d < -1e-12
            row['improved_slots'], row['worsened_slots'] = wins, losses
        table[name] = row

    seeds = [f'reference_seed_{seed}' for seed in [7, 17, 27]]
    spread = {}
    for metric in metric_config:
        values = [table[name]['results'][PROPOSED][metric] for name in seeds]
        spread[metric] = {'values': values, 'min': min(values), 'max': max(values),
                          'range': max(values)-min(values), 'population_sd': statistics.pstdev(values)}
    for scale_type in ['scientific_cohort_normalized_delta', 'audit_cohort_normalized_delta']:
        values = [table[name][scale_type] for name in seeds]
        spread[scale_type] = {'values': values, 'min': min(values), 'max': max(values),
                              'range': max(values)-min(values)}
    summary = {
        'reference': reference, 'runs': table, 'seed_spread': spread,
        'scientific_cohort_scales': quality_scales, 'audit_cohort_scales': all_scales,
        'normalization_note': 'Illustrative population standard deviations over the recorded audit '
            'runs, per method/metric. Actual future-attempt variances are not available. '
            'Do not treat these deltas as an absolute official rank. Missing slots contribute zero '
            'per the stated rank contract; failed and nonfinite runs are not ranked.',
    }
    (BASE / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['case\thidden_rc\tPSNR\tSSIM\tLPIPS\tFVD\tforeground_MAE\tedge_MAE\toverhead\taudit_delta']
    for name, row in table.items():
        values = row.get('results', {}).get(PROPOSED, {})
        cells = [name, str(row['hidden_rc'])]
        for key in ['psnr','ssim','lpips','fvd','foreground_mae','edge_mae','inference_overhead']:
            value = values.get(key)
            cells.append(f'{value:.9g}' if isinstance(value, (float, int)) else 'MISSING')
        value = row.get('audit_cohort_normalized_delta')
        cells.append(f'{value:+.9g}' if value is not None else 'UNRANKED')
        lines.append('\t'.join(cells))
    (BASE / 'summary.tsv').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    print('SEED SPREAD:', json.dumps(spread, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
