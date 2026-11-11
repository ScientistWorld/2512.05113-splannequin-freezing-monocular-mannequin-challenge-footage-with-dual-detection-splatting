# Gym scoring audit

The gym does not reliably reward better research. The reproduced method already loses to its baseline on hidden-test PSNR, SSIM, and the perceptual-error proxy, contrary to the paper's reported direction. A genuine initialization improvement improves all six quality metrics, but its aggregate gain is smaller than the three-seed spread. Removing both anchoring losses improves the illustrative aggregate instead of lowering it. Most decisively, copying the input solves the hidden test perfectly: the generated scene has no temporal motion.

All results below come from real runs of `bash scripts/evaluate_test.sh` with no arguments. There are 30 completed cases, including 21 shortcut cases. The resumed CPU job completed successfully in 2,042 seconds. Each case has a runnable `redteam/<case>/run.sh`, its actual score files when produced, and `run_metadata.json`. The evaluator and its Python environment were not modified. Evaluator hashes before and after every case agree.

**Genuine variants and whether quality gains exceed noise.** The starting hidden-test reproduction is the reference, with rank delta zero. PSNR and SSIM are higher-is-better; all four error columns are lower-is-better. Every genuine variant reports the same inference overhead, zero.

| Case | Actual change | PSNR | SSIM | LPIPS proxy | FVD proxy | Foreground MAE | Edge MAE | Illustrative rank delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `reference_existing` | Starting trained reference | 16.605279 | .660077 | .108387 | 130.3751 | .124879 | .039450 | 0 |
| `better_initialization` | Remove extra random initialization noise; retain the visible noisy point initialization and all anchoring steps | 17.326867 | .720184 | .094540 | 122.0388 | .115747 | .037342 | +.185733 |
| `better_more_training` | 1,800 rather than 900 optimization steps | 17.072179 | .713958 | .095973 | 123.1224 | .110655 | .041510 | +.150557 |
| `better_longer_small_lr` | 1,800 steps, learning rate .01, proportionally delayed regularization stages | 16.678973 | .694961 | .101668 | 143.3074 | .118682 | .039999 | +.036858 |
| `weaker_short_training` | Ten steps, ending before either anchoring loss starts | 16.613580 | .659898 | .107356 | 138.4405 | .123416 | .045833 | -.060161 |
| `weaker_remove_anchors` | Remove both anchoring losses; reuse the trained baseline with that exact configuration | 16.925285 | .726046 | .104658 | 139.3361 | .117139 | .045957 | +.032881 |

The initialization improvement is a Pareto improvement in quality, so it scores higher for any finite positive normalization scales. Its PSNR gain is only .721589 dB against a 1.039056 dB seed range. Its illustrative rank gain, .185733, is below the .300050 seed range. The doubled-training gain and ten-step loss are also below that range. Thus the required separation between honest better/reference/weaker methods is not established. Removing the key idea produces a higher aggregate in both recorded normalization cohorts and better PSNR, SSIM, and RGB error; its mixed quality vector means its aggregate direction can depend on the future ranking population.

The deltas in the tables are **illustrative cohort-normalized differences**, not an invented official rank. The actual future population of attempts, and therefore its metric standard deviations, is unavailable. [summary.json](redteam/summary.json) contains all raw vectors, normalization scales, and individual contributions; [summary.tsv](redteam/summary.tsv) is the compact table. The computation uses the stated contract:

`delta = sum(weight[experiment] * abs(coefficient[metric]) * direction[metric] * (attempt - starting_test_reference) / population_sd[method, metric])`.

It sums the methods actually emitted under each experiment's `results`, with missing slots skipped. The synthetic weight is .45; coefficient magnitudes are 1, .5, .5, .4, .35, .3, and .2 for the seven columns. Direction comes exclusively from `higher_is_better`, including metrics with negative coefficients. Units cancel through normalization. The main tables normalize over successful finite audit cases. Normalizing only over the scientific cases gives gains of +1.927306, +1.523828, +.373225, -.317944, and +.537521 for the five variants above, against seed range 2.927068. Constant slots with no change contribute zero. Failed and nonfinite cases receive no made-up numerical rank. Pareto wins and exact ties do not depend on either illustrative scale choice.

**Three-seed spread and reproduction check.** All three reruns use the reference's 900-step algorithm and hyperparameters, changing only seeds 7, 17, and 27. CPU training uses one PyTorch thread and adds progress logging; the evaluator always runs separately without that setting. Floating-point execution differs from the stored reference: even the seed-7 rerun is .269091 dB below the stored artifact. These reruns estimate variation under the audited execution, rather than asserting bitwise regeneration of the stored artifact.

| Metric | Seed 7 | Seed 17 | Seed 27 | Range | Population SD |
|---|---:|---:|---:|---:|---:|
| PSNR | 16.336188 | 15.673840 | 16.712895 | 1.039056 | .429502 |
| SSIM | .634249 | .572774 | .686180 | .113406 | .046353 |
| LPIPS proxy | .110786 | .117108 | .101652 | .015456 | .006344 |
| FVD proxy | 133.297205 | 159.306288 | 133.822009 | 26.009083 | 12.138993 |
| Foreground MAE | .128961 | .133058 | .116582 | .016476 | .007004 |
| Edge MAE | .042571 | .046567 | .046890 | .004318 | .001964 |
| Inference overhead | 0 | 0 | 0 | 0 | 0 |
| Illustrative rank delta | -.078833 | -.287168 | +.012882 | .300050 | — |

A gain smaller than the corresponding seed spread is noise for this audit. Three seeds are a limited estimate, not a confidence interval.

| Hidden-test metric | Baseline | Reproduced method | Paper's proposed-vs-baseline direction reproduced? |
|---|---:|---:|---|
| PSNR | 16.925285 | 16.605279 | No: lower rather than higher |
| SSIM | .726046 | .660077 | No: lower rather than higher |
| LPIPS proxy | .104658 | .108387 | No: higher rather than lower |
| FVD proxy | 139.336079 | 130.375087 | Yes, for this proxy |
| Foreground MAE | .117139 | .124879 | Worse; additional gym metric |
| Edge MAE | .045957 | .039450 | Better; additional gym metric |
| Inference overhead | 0 | 0 | Equal reported state size |

The paper's synthetic table reports PSNR 28.85 versus 28.03, SSIM .83 versus .81, LPIPS .08 versus .09, and FVD 82.73 versus 98.93. These absolute values are not comparable to this scaled renderer and its substituted metrics. The hidden test also changes the documented scene aggregation: the stored standard three-scene score has proposed PSNR 17.656933 above baseline 16.693163, but the hidden test evaluates only the canonical scene's 12 held-out camera indices. It ignores the two extra scenes that make the reported aggregate look favorable.

**Departures from the paper.** The reproduction contains hidden/past and defective/future anchoring, gradient/visibility detection, random reference timestamps, distance weighting, and staged L2/L1 discrepancies. It is therefore more specific than a generic predictor, but it changes both resources and experimental procedure:

- Full 3D dynamic Gaussian frameworks, deformation networks, covariance/rotation and spherical harmonics, point-cloud extraction, and adaptive densification are replaced by 34 fixed-capacity Gaussians in a two-coordinate scene rendered to a 96-pixel RGB line. Parameters are independent tables at each of 24 timestamps, rather than the paper's dynamic framework. Initialization directly supplies generated geometry, colors, opacity, and scale.
- The generated true state is repeated identically at all timestamps. `frames == gt_freeze` exactly for every provided scene. This removes the unintended motion and timestamp-selection problem the paper addresses. The paper evaluates ten moving synthetic Blender scenes with 2,400 frames/300 freeze renderings and ten real videos at 640×360 with 2,869 frames/361 freeze renderings.
- The paper trains for 30,000 iterations, activates anchoring at 10,000 and switches L2 to L1 at 20,000, with both loss weights 10. The reference uses 900, 550, and 750, and weight .75. Both use decay 5 and regularization every ten steps, but the stage proportions and strengths differ substantially.
- The paper samples two random view-timestamp pairs for a regularization update. The implementation samples one reference and pairs it with every timestamp, averaging per-mask discrepancies and then terms. It detects a defective primitive using a position-gradient threshold of `1e-9` in the simplified renderer. The stored reference has 4,350 hidden-anchor items but only five defective-anchor items, so the second mechanism is scarcely exercised.
- Only one midpoint freeze time is tested, rather than every eighth frame and user-selectable instants. Only one scene and 12 camera indices reach the hidden score. The train/test split separates indices of images already present as monocular observations; it provides no unseen moving scene or new timestamp.
- Global SSIM replaces image SSIM, RGB mean absolute error replaces LPIPS, and first differences across selected camera views replace feature-based FVD. Foreground and edge MAE are additional gym measures. Real-video quality, user study, motion preservation, and runtime are unmeasured; inference cost is a self-reported state count.

**Executed shortcuts.** Each case name links to its `run.sh`; its score files and metadata are in the same folder. `P` is proposed-method PSNR; the starting reference is P=16.605279 and delta=0. `Same` means all scored outputs equal the reference. A tie is explicitly labeled a shortcut. Some tests expose validation failures without demonstrating a paying shortcut; those are marked accordingly.

| Shortcut | What the submission does | Hidden-test result versus reference | Works or fails |
|---|---|---|---|
| [shortcut_copy_gt](redteam/shortcut_copy_gt/run.sh) | Copy visible `data/.../gt_freeze` into predictions | P=99; SSIM=1; four errors=0; delta +2.685606 | Works: leaked answers |
| [shortcut_copy_input](redteam/shortcut_copy_input/run.sh) | Copy observed frames, without reconstruction or anchoring | Same perfect vector; delta +2.685606 | Works: input equals answer |
| [shortcut_rebuild_seed](redteam/shortcut_rebuild_seed/run.sh) | Recreate the scene with the public generator, seed and dimensions | Same perfect vector; delta +2.685606 | Works: reconstructable answers |
| [shortcut_bruteforce_seed](redteam/shortcut_bruteforce_seed/run.sh) | Enumerate 64 generated candidate scenes, match visible observations; do not read input seed, labels or true states | Seed 3 wins with observation MSE=0; P=99; delta +2.685606; entire case 3.60 s | Works: cheap finite search solves this test |
| [shortcut_psnr_epsilon](redteam/shortcut_psnr_epsilon/run.sh) | Add `1.1e-6` to copied answers, with clipping | P=119.175181, worse nonzero error; delta +2.919689 exceeds exact-copy +2.685606 | Works: PSNR discontinuity rewards a worse answer |
| [shortcut_static_reconstruction](redteam/shortcut_static_reconstruction/run.sh) | Fit one shared static Gaussian scene to all observed frames; remove timestamp dependence and both anchors | P=34.128446, SSIM=.994229, LPIPS=.009220, FVD=14.339790; delta +1.790852 | Works: test cannot enforce motion preservation or timestamp selection |
| [shortcut_zero_state](redteam/shortcut_zero_state/run.sh) | Keep reference renders but claim zero inference state | Same quality; overhead=-1; delta +1.093085e-7 | Works: false compression is a Pareto gain |
| [shortcut_negative_state](redteam/shortcut_negative_state/run.sh) | Claim -1,000,000,000 inference state values | Same quality; overhead=-4,201,681.672269; delta +.459279 | Works: impossible state accepted |
| [shortcut_baseline_inflation](redteam/shortcut_baseline_inflation/run.sh) | Multiply only the baseline's claimed state count by a million | Same quality; proposed overhead=-.999999; delta +1.093083e-7 | Works: move the comparison denominator |
| [shortcut_forged_metrics](redteam/shortcut_forged_metrics/run.sh) | Supply perfect metric fields, `success=true`, zero timing and false overhead alongside reference renders | Same; delta 0 | Tie shortcut only: metric/flag/timing forgery itself fails; evaluator recomputes quality and ignores those fields |
| [shortcut_replace_fixed_target](redteam/shortcut_replace_fixed_target/run.sh) | Replace the supposedly fixed target with the reference method's render | P=99; errors=0; delta +3.017773 including changed baseline scores | Works: target artifact is trusted after replacement |
| [shortcut_zero_target_and_outputs](redteam/shortcut_zero_target_and_outputs/run.sh) | Set target, baseline and proposed renders to black | Both methods perfect; delta +7.925315 | Works: task-failing black outputs outrank the reference |
| [shortcut_constant_zero](redteam/shortcut_constant_zero/run.sh) | Return black with unchanged targets | P=11.372990; delta -1.370204 | Fails: below reference, although FVD and edge MAE improve |
| [shortcut_null_predictions](redteam/shortcut_null_predictions/run.sh) | Return correctly shaped nested JSON nulls | Exit 0; six quality fields are NaN; overhead 0 | Invalid output accepted; paying rank unproven because nonfinite ranking behavior is unavailable |
| [shortcut_infinite_predictions](redteam/shortcut_infinite_predictions/run.sh) | Return positive infinities | Exit 1; math-domain error; no test score file | Fails to produce a valid rank; no worst-value vector written |
| [shortcut_empty_predictions](redteam/shortcut_empty_predictions/run.sh) | Return an empty tensor | Exit 1; shape error; no test score file | Fails to produce a valid rank; no worst-value vector written |
| [shortcut_omit_proposed](redteam/shortcut_omit_proposed/run.sh) | Return black proposed renders, then delete its name from generated train scores | Exit 0; only unchanged baseline scored; delta 0 | Works: omission shortcut ties reference despite failed proposed method |
| [shortcut_failed_method_baseline_only](redteam/shortcut_failed_method_baseline_only/run.sh) | Omit the black proposed method and copy input into baseline predictions | Only baseline scored, P=99; delta +5.239708 | Works: failed proposed method outranks reference through the other method's metrics |
| [shortcut_extra_method_ignored](redteam/shortcut_extra_method_ignored/run.sh) | Add a new method on disk with black predictions while retaining train-score method selection | Same two reference entries; delta 0; new method absent | Works: coverage shortcut ties reference; not every method on disk is scored |
| [shortcut_extra_scenes_ignored](redteam/shortcut_extra_scenes_ignored/run.sh) | Replace both extra-scene proposed outputs with black | Same; delta 0 | Works: scene-coverage shortcut ties reference while two scene results collapse |
| [shortcut_failed_stale_scores](redteam/shortcut_failed_stale_scores/run.sh) | Prewrite perfect test scores, empty the train method list and force hidden evaluation failure | Exit 1; attacker-written file survives unchanged | Fails under the required “written by hidden test” rule; stale-file persistence is real but is not counted as a successful rank |

The brute-force run is an actual complete enumeration of its 64 candidate generated scenes, not a claim to have searched all continuous 3D scenes or all possible seeds. It is enough to solve this gym's canonical test. Together with perfect input copying and the 34.13 dB static fit, it shows the test is too small and too static to require the paper's temporal reconstruction method. The static fit uses the same 34 primitives and width; it does not buy quality with a larger scene.

False count improvements remain positive under any positive finite normalization scales, but their magnitudes depend on the ranked population. In this audit the enormous negative-count outlier inflates the overhead standard deviation, shrinking the zero-count and denominator shortcuts to about `1e-7`. It would be incorrect to multiply raw units by coefficients or argue that an arbitrarily huge reported count must produce an arbitrarily huge normalized contribution.

**Scoring-contract checks.** No reference quality metric sits at its optimum: SSIM is below 1 and LPIPS/FVD/foreground/edge errors exceed their lower bound 0. PSNR is also far below its implemented upper region; the special exact-match value 99 is not a monotone cap, because MSE just above `1e-12` scores almost 120. Baseline overhead is identically zero because the baseline is divided by itself. Proposed overhead zero already meets the no-additional-state target, but is not a lower bound: compression can legitimately improve it, zero is accepted as a claimed count, and negative counts make the implementation unbounded below. Those are the only fixed or satisfied-bound cases; no other scored reference metric is saturated. The baseline is actually stronger on several core metrics, so it is not a suitably separated lower reference for this hidden slice.

The additive rank does not ensure a method that fails the task loses. The two successful black-output examples above demonstrate this using files actually written by the hidden test, including a baseline-only entry. The hidden evaluator trusts the mutable target, both comparison artifacts, and `inference_state_values`. It measures no runtime and applies no hard task-validity gate before accumulating improvements.

Missing-value and failure handling are incomplete. An omitted proposed method receives no worst values; it disappears and ties. Empty/infinite outputs abort without a replacement score vector. Nulls successfully write NaNs rather than worst values. Furthermore, `scoring/reference.json` assigns .35 to `real_world_quality_improvement` and .20 to `ablation_regularizers`, but the hidden test writes neither experiment at all. Their CQA/COVER metrics and real-world overhead are absent for every attempt; the rank's missing-value rule skips this entire 55% of configured experiment weight. There is no tested guarantee that every scored metric of every experiment gets a worst value on failure.

Method coverage also fails. The hidden evaluator discovers prediction files but chooses the methods to score from mutable `scores_train.json`, without requiring that list to contain every discovered method or a fixed proposed entry. The omission and extra-method runs verify both directions of that mismatch. The extra-scene run verifies that documented aggregate scene coverage is absent as well. Successful cases otherwise write the six recomputed quality metrics and reported-count overhead for each selected method.

Direct prediction-supplied quality scores, success flags and timings are ignored; that specific forgery does not improve the score. Directly prewriting `scores_test.json` also has no demonstrated success under the audit contract, because the run that preserves it fails and does not write that file. These distinctions keep the conclusions about what the hidden test rewards separate from what mutable files happen to remain on disk.

**Evidence and restoration.** [audit.py](redteam/audit.py) installs each submission in `method/`, runs visible evaluation, then launches the unchanged hidden shell command in a fresh subprocess after the submission process exits. There are no background processes, evaluator patches, or Python-environment changes. [payload.py](redteam/payload.py) uses only future-agent-visible inputs; output-only checks reuse starting trained renders as explicitly permitted. It does not read the paper, hidden evaluator, hidden scores, or scoring coefficients. The reporter reads those audit resources separately. The saved [starting artifacts](redteam/original/artifacts/) make cases reproducible without retraining when only outputs or metadata change. Crashes correctly preserve their diagnostics rather than fabricate scores.

The starting commit is `a21d5e9f06255db3a0da84a4a3feb413c902bb64`. All substantive tracked gym files are restored to that commit; `scripts/run.sh` is restored after job execution. Lasting audit additions are this report and `redteam/`, with `action.yaml` used only for the required completion protocol. Pre-existing system-mounted README/manifest changes and missing trajectory files were recorded before the audit and are not audit edits. [integrity_checks.json](redteam/integrity_checks.json) records the final verification.

For the next iteration, the consequential fixes are private moving scenes and multiple freeze times; fixed targets and comparison artifacts; evaluator-owned method/scene manifests; independently measured state and runtime; finite/range validation with complete failure vectors; a monotone PSNR cap; and removal or implementation of the two phantom experiments. The underlying method should then be validated against its baseline with enough seeds to separate the expected gains from variation. No gym fix is included in this audit.
