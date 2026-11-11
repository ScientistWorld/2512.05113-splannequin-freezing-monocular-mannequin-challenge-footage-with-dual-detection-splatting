#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py --shortcuts
python redteam/summarize.py
python - <<'PY'
import hashlib
import json
from pathlib import Path

base = Path('redteam')
before = json.loads((base / 'visibility_before.json').read_text())
changes = []
for case, expected in before.items():
    path = base / case / 'scores_test.json'
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    if actual != expected:
        changes.append(case)
    metadata = json.loads((base / case / 'run_metadata.json').read_text())
    assert metadata['submission_writes_minimal_method_source'], case
    assert metadata['method_source_at_test']['experiments']['synthetic_validation_paper']['results'] == {
        '4dgaussians': {}, '4dgaussians_splannequin': {}}, case
report = {'shortcut_cases_rerun': len(before), 'changed_score_files': changes,
          'all_shortcuts_explicitly_create_visible_method_list': True}
(base / 'visibility_result.json').write_text(json.dumps(report, indent=2) + '\n')
print('VISIBILITY CHECK:', json.dumps(report), flush=True)
PY
