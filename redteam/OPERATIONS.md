The first batch (job `5486f34e-9af`) was cancelled after 40 minutes without
finishing its first 900-step training run. Its starting-reference evaluation
completed and is retained. The resumed batch limits PyTorch training to one CPU
thread and logs training loss every 100 iterations. It does not change the
training algorithm or the evaluator's process or environment.

`bash redteam/run_all.sh` resumes completed cases and runs all remaining ones.
Each case also has its own `run.sh`. The driver restores the starting data,
scores, and prediction artifacts before each case, installs the submission in
`method/`, runs it, runs the visible evaluator, and invokes
`bash scripts/evaluate_test.sh` with no arguments as a fresh subprocess.
The evaluator files are never edited, and before/after hashes are recorded.
The driver restores the starting files in a `finally` block. A hard job kill
can interrupt that cleanup; `python redteam/audit.py --restore` repairs it.

`original/artifacts/` contains small starting trained renders, reused for the
metadata-only and output-only checks as permitted by the audit instructions.
The payload does not read the audit's original test scores, hidden evaluator,
paper, or scoring coefficients. Those are read only by the audit driver and
reporter. All real training runs use the visible method and scene.

The population of future ranked attempts is unknown. `summary.json` records all
raw metrics and two explicitly illustrative normalized deltas, using the given
absolute coefficients, metric directions, and experiment weights. One uses
only the scientific variants for normalization; the other uses the successful
finite audit runs. Pareto improvements, ties, and omitted-slot findings do not
depend on these illustrative normalization scales. Nonfinite and failed runs
are not assigned an invented rank.
