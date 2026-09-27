# Person 2: XGBoost reproduction and reverse-union evaluation

## Handoff (2026-09-27)

Branch `codex/reverse-model-evaluation` starts at `05de9a6`. Person 1's
branch and the existing production bundles are unchanged.

**Forward reproduction is complete. Reverse comparison is pending the completed
Person 1 union artifacts.** The four supplied files match the SHA-256 values in
`artifacts/reports/person1_feature_ablation.json` exactly. All eight reproduced
configurations match the earlier calibration/development metrics and thresholds
exactly, despite Python patch and transitive dependency differences.

These are exploratory results on the historical 3,000 training / 1,000
calibration / 1,000 development S1 groups. Precision and recall are micro link
metrics; F0.5 is macro over all S1s, including singletons. `validation` in the
frozen split file means development. No independent confirmation, new test
accuracy, leaderboard result, or production promotion is claimed.

| Forward model | Threshold (calibration) | Calibration F0.5 | Development F0.5 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 16 features | 0.645 | 0.886242 | 0.889840 | 0.966655 | 0.787757 |
| 47 features | 0.600 | 0.906297 | 0.909947 | 0.971245 | 0.821224 |

The 47-feature configuration also wins on calibration. Its exploratory paired
improvement over the 16-feature model is +0.020107 macro F0.5, with the existing
2,000-resample S1 bootstrap interval [0.012622, 0.027676]. This interval does not
account for repeated model exploration or domain shift.

| Split | S1s | Candidates | Found / true links | Candidate recall |
|---|---:|---:|---:|---:|
| Training | 3,000 | 74,658 | 9,629 / 10,460 | 0.920554 |
| Calibration | 1,000 | 25,399 | 3,266 / 3,531 | 0.924950 |
| Development | 1,000 | 24,758 | 3,279 / 3,496 | 0.937929 |

Total candidates: **124,815**. Development candidates per S1: median 25,
p95/p99/max 40. There are 217 unretrieved true development links, across 170 S1s.
They are included in recall denominators and never inserted into training pools.

Useful development slices (macro F0.5):

| Slice | S1s | 16 features | 47 features | Candidate recall |
|---|---:|---:|---:|---:|
| India | 404 | 0.859266 | 0.886578 | 0.908291 |
| US | 596 | 0.910564 | 0.925787 | 0.956868 |
| No true matches | 52 | 0.923077 | 0.923077 | N/A |
| One true match | 59 | 0.772128 | 0.824859 | 0.932203 |
| Two or more | 889 | 0.895708 | 0.914826 | 0.938027 |
| Any candidate missing address | 220 | 0.852742 | 0.886127 | See pair/source diagnostics |
| Forward pool misses any truth | 170 | 0.745442 | 0.786146 | 0.686416 |

There are no non-Latin S1 names or missing S1 addresses in this development
sample; empty slices have null metrics. S2/S3 link metrics are in the JSON report.
Source-specific macro metrics use source-restricted truth, so they are diagnostic
and differ from the overall competition metric.

### Measured runtime

All measurements are CPU, two threads, on this local arm64 macOS machine.
For all 124,815 candidates, standalone saved-model verification took **2.63 s**
for 16-feature extraction and **8.49 s** for 47-feature extraction, plus **0.066 s**
and **0.055 s** respectively for prediction. The original training run took
**0.254 s / 0.441 s** to fit these models. Its shared 47-column matrix took 8.27 s.
These are component timings, excluding retrieval, threshold search, and I/O;
there is no end-to-end production runtime claim. The report retains three repeated
10,000-pair feature benchmarks for every configuration.

### Artifacts and versions

- Tracked results: `artifacts/reports/person2_forward_reproduction_v1.json`.
- Local models: `artifacts/experiments/person2_forward_reproduction_v1/`.
  Each configuration has native XGBoost `model.json`, fitted `extractor.json`,
  integrity-checked `bundle.json`, and `threshold_sweep.json`. The two main controls
  also have saved pair scores. Models, vocabulary and pair scores remain ignored
  by Git under the existing artifact policy; reproduce or transfer privately.
- Lexical feature version: `matching-features-v1` (unchanged).
- New feature version: `target-competition-v1` (12 columns).
- New model bundle version: `reverse-union-xgb-v1` (47 or 59 columns).
- XGBoost 3.2.0, NumPy 2.4.3, pandas 3.0.1, scikit-learn 1.8.0,
  RapidFuzz 3.14.6, AnyAscii 0.3.3; Python 3.14.6 for this reproduction.
  The dependency file pins the measured environment, including retrieval tests.
- Baseline 16-feature model SHA-256:
  `2e69ca314c2031aa53b51c8d2a2f77e22bc7559dcacf3dce01d14908b1e403b3`.
- Baseline 47-feature model SHA-256:
  `d4bbab826f7c6e57edd03b548db300e92046dc59493c5130b3c89acf31978a1f`.

## Fixed comparison protocol

The original XGBoost settings are retained: 200 trees, depth 4, learning rate
0.08, L2 regularization 1, CPU histogram method, binary logistic objective,
logloss metric, seed 2026, two threads. There is no early stopping or internal
pair split. The comparison reads these settings from the reproduced baseline
report and checks its bundle, input hashes, training IDs, and reproduced metrics.
The vocabulary is fitted only on training S1 text, with no target or held-out
text used for fitting.

Prespecified runs:

1. Saved forward 47-feature baseline.
2. Forward pool with 47 + 12 competition features, using the full top-8 context.
3. Retrained union K=1, K=3, K=8 with the original 47 features.
4. Retrained union K=1, K=3, K=8 with 47 + 12 competition features.

The forward control separates competition evidence from added candidates. Each
union's 47-feature control separates candidate-pool effects from feature effects.
All candidates survive into scoring; there is no new forward cap or truth injection.

The 12 extra columns, in order, are forward score/rank; reverse score/rank;
best/second owner scores; gap to the best owner; margin over the best other owner;
reverse-rank censor flag; forward/reverse route flags; and whether the query is
the best reverse owner. Missing numeric evidence stays NaN. A censored reverse
score/rank is never zero or K+1. Negative margins and zero tie margins are valid.
Route flags come from explicit route membership, not rank availability. No owner
ID is used as a categorical feature, and retained-context length is not used as
total competitor count. Forward metadata and text are checked against the frozen
raw cache; aliases must agree exactly. Reverse-only forward metadata stays null.

Thresholds use the existing calibration-only coarse grid (0.30–0.95, step 0.05,
plus accept-all/reject-all), then a 0.005 local refinement. Ties favor the higher
threshold. The comparison's model/K choice uses calibration macro F0.5 only;
development values and paired bootstrap intervals are exploratory reporting.
No production update is performed even if this exploratory comparison improves.
Fresh confirmation and full inference/runtime validation remain necessary.

Every run saves metrics, slices, candidate counts/recall, route counts, training
class counts, predictions, threshold sweep, feature order, bundle/input hashes,
source hashes, dependency versions, and feature/fit/prediction/calibration runtime.
The top-level report carries Person 1's retrieval manifest and union build time
separately from model runtime, plus a completion marker. Partial outputs cannot
be mistaken for completed comparisons.

## Reproduce

Run from this branch's checkout. Input paths may point outside the checkout.
Fresh output paths are required; do not overwrite existing model directories.

```bash
python3.14 -m venv .venv-model
.venv-model/bin/python -m pip install -r requirements-model-evaluation.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# On macOS, use one OpenMP runtime for sklearn, XGBoost and sparse-dot-topn.
# Omit this on other operating systems.
export DYLD_LIBRARY_PATH="$PWD/.venv-model/lib/python3.14/site-packages/sklearn/.dylibs"

.venv-model/bin/python -m pytest -q

INPUT_DIR='/Users/parthbhat/Downloads'
BASELINE='artifacts/experiments/person2_forward_reproduction_v1'
.venv-model/bin/python scripts/24_feature_ablation.py \
  --sample "$INPUT_DIR/sample_s1.jsonl" --pairs "$INPUT_DIR/pairs.jsonl" \
  --split "$INPUT_DIR/split.json" --allowed "$INPUT_DIR/candidate_pairs_internal.tsv" \
  --output "$BASELINE"

.venv-model/bin/python scripts/28_reverse_model_evaluation.py summarize-forward \
  --sample "$INPUT_DIR/sample_s1.jsonl" --pairs "$INPUT_DIR/pairs.jsonl" \
  --split "$INPUT_DIR/split.json" --allowed "$INPUT_DIR/candidate_pairs_internal.tsv" \
  --baseline "$BASELINE" --output artifacts/experiments/reproduced_forward_summary.json
```

The baseline command runs the original eight feature ablations unchanged. It
records a historical development-best field; the summary adds calibration-only
selection, which chooses the same 47-feature configuration in this reproduction.

### Once Person 1's full scan finishes

Point `UNIONS` to the directory containing **all three** `union_k1.jsonl`,
`union_k3.jsonl`, `union_k8.jsonl`, and the producer's **complete `report.json`**.
The comparison verifies full-scan completion, provenance hashes, nested pools,
forward preservation, route constraints, unknown encodings and file hashes before
creating its output. Do not manually synthesize a manifest for a partial scan.
If the producer is on another machine, these ignored artifacts must be transferred
privately; a Git pull does not include them.

```bash
UNIONS='artifacts/experiments/reverse_person1_full_v1/unions'
.venv-model/bin/python scripts/28_reverse_model_evaluation.py compare \
  --sample "$INPUT_DIR/sample_s1.jsonl" --pairs "$INPUT_DIR/pairs.jsonl" \
  --split "$INPUT_DIR/split.json" --allowed "$INPUT_DIR/candidate_pairs_internal.tsv" \
  --baseline "$BASELINE" --unions "$UNIONS" \
  --output artifacts/experiments/person2_reverse_comparison_v1
```

After a successful complete comparison, copy its compact report into
`artifacts/reports/`, update this handoff with the calibration-selected comparison
and all development metrics, then commit/push this branch. Keep model artifacts
and input data outside Git. No reverse results are recorded yet.

Bundle inference on a finalized union uses the same extractor and feature order:

```bash
.venv-model/bin/python scripts/28_reverse_model_evaluation.py score \
  --bundle 'artifacts/experiments/person2_reverse_comparison_v1/union_k3_47+competition' \
  --sample "$INPUT_DIR/sample_s1.jsonl" --pairs "$UNIONS/union_k3.jsonl" \
  --output artifacts/experiments/reverse_k3_scores.tsv
```

This produces diagnostic pair scores, not a production submission. Supply union
metadata for competition models and the matching retrieval configuration.

## Validation

The added tests cover missingness, explicit routes, ties/negative margins,
invalid ranks, label-independent pools, no truth insertion, partial scans,
changed hashes, changed forward ranks, dropped edges, development-label leakage,
serialization/inference parity, tampered bundles, and an end-to-end fixture that
retrains all seven prespecified comparison configurations. The complete test
suite includes the existing reverse retrieval/resume/union tests.

Measured validation: **71 tests passed in 5.79 seconds**. `pip check` found no
broken requirements. On this macOS environment the first suite invocation
stalled when multiple OpenMP copies were loaded; the shared-library setting in
the commands above resolved it. The baseline reproduction itself completed
without that override and matched all historical metrics exactly.
