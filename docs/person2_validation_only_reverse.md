# Validation-only reverse comparison: reverse_dev1000_v1

## Scope and conclusion

This is a **frozen-model candidate-pool diagnostic on the original 1,000 validation
S1s only**. The original 3,000 training S1s, 1,000 calibration S1s, forward pools,
models, feature extractors, and calibration-selected thresholds are unchanged.
Both original calibration metrics and forward validation metrics reproduce exactly.
No model fitting, threshold search, K selection, or production promotion occurred.

The reverse pools improve candidate recall, but the frozen 47-feature model loses
precision and macro F0.5. These results do not establish how a model retrained on
reverse candidates with competition features would perform. That experiment needs
completed training and calibration unions and remains a separate pending task.
All reported validation results are exploratory.

## Provenance checks

The new explicit `validation-only` command reconstructs the producer's subset
inputs from the original frozen files and requires exact SHA-256 agreement:

- Sample: original-order validation lines, retaining their bytes, including labels.
- Forward cache: original-order raw lines for finalized validation edges only.
- Forward TSV: original-order validation records, using the producer's CSV writer
  with CRLF line endings (the original full TSV uses LF).
- Split: `{"validation": <original validation ID list>}`, JSON indent 2 plus newline.

These deterministic projections match all four uploaded report hashes. This is
an exact verification of the subset's input files, not a replacement of the
producer's hashes with full-split hashes. The standalone `manifest.json` hash
must match the producer report and its content must equal the embedded manifest.
The manifest records 1,000 emitted S1s, all 2,206,821 S1 competitors, a completed
10,320,219-target scan, and no prefix benchmark.

The existing checks still enforce union file hashes, nested pools, valid routes,
unknown reverse evidence, preserved forward text/scores/ranks and aliases, and
no dropped forward edges. Only owners in the original validation split are
accepted. The full-split comparison continues to reject these subset artifacts.
Neither candidate validation nor subset projection inserts ground-truth edges.

The output explicitly records `scope.mode=validation-only`,
`full_split_comparison=false`, `model_retrained=false`,
`threshold_reselected=false`, `k_selected=false`, full and subset hashes,
validation ID hash, frozen model hashes, source hashes, runtime, and slices.
Competition metadata is preserved and validated; the frozen 16/47-feature models
do not consume it. A 59-feature model is not trained from validation data.

## Results

All candidate statistics below concern the same 1,000 validation S1s and 3,496
true links. Precision/recall are micro link metrics; macro F0.5 includes singletons.

| Pool | Candidates | Candidate recall | Recovered true links | Lost forward edges |
|---|---:|---:|---:|---:|
| Forward | 24,758 | 0.937929 | 0 | 0 |
| Union K=1 | 26,963 | 0.972540 | 121 | 0 |
| Union K=3 | 43,865 | 0.974256 | 127 | 0 |
| Union K=8 | 76,132 | 0.974256 | 127 | 0 |

**Frozen 47-feature XGBoost**, threshold **0.600**:

| Pool | Macro F0.5 | Precision | Recall | F0.5 change |
|---|---:|---:|---:|---:|
| Forward | 0.909947 | 0.971245 | 0.821224 | — |
| Union K=1 | 0.900903 | 0.940575 | 0.842105 | -0.009043 |
| Union K=3 | 0.898848 | 0.934686 | 0.843249 | -0.011098 |
| Union K=8 | 0.897475 | 0.931438 | 0.843249 | -0.012472 |

The exploratory paired 95% S1 bootstrap interval for K=1's F0.5 change is
[-0.015756, -0.002471]. The JSON report includes all intervals; these do not account
for repeated use of the historical validation set or multiple comparisons.

**Frozen 16-feature control**, threshold **0.645**:

| Pool | Macro F0.5 | Precision | Recall |
|---|---:|---:|---:|
| Forward | 0.889840 | 0.966655 | 0.787757 |
| Union K=1 | 0.892218 | 0.953544 | 0.804348 |
| Union K=3 | 0.890797 | 0.949713 | 0.804920 |
| Union K=8 | 0.889278 | 0.946200 | 0.804920 |

Its K=1 improvement is small and inconclusive: +0.002378, exploratory interval
[-0.002841, 0.008152]. No model or K is selected using these validation outcomes.

### Fixed validation slices: 47-feature macro F0.5

| Slice | S1s | Forward | K=1 | K=3 | K=8 |
|---|---:|---:|---:|---:|---:|
| India | 404 | 0.886578 | 0.889269 | 0.887063 | 0.885335 |
| US | 596 | 0.925787 | 0.908790 | 0.906838 | 0.905704 |
| Singletons | 52 | 0.923077 | 0.884615 | 0.884615 | 0.884615 |
| One true match | 59 | 0.824859 | 0.801246 | 0.790816 | 0.789282 |
| Multiple true matches | 889 | 0.914826 | 0.908470 | 0.906851 | 0.905408 |
| Forward missed any truth | 170 | 0.786146 | 0.832209 | 0.832709 | 0.828773 |

Reverse retrieval helps the original retrieval-miss slice but hurts US and
singleton performance under the unchanged classifier policy. Candidate-address
missingness and added-candidate slices in the JSON report can have different
membership across pools and should not be read as fixed-cohort comparisons.

### Runtime and artifacts

The complete validation diagnostic, including input checks, calibration
verification, both models, all pools, bootstrap reporting and score output, took
**21.72 seconds**. For the 47-feature model, feature extraction/prediction seconds
were forward **1.670/0.011**, K=1 **1.564/0.013**, K=3 **2.638/0.020**, and K=8
**4.733/0.035**. These sequential measurements can reflect cache effects; they
exclude retrieval. The producer reports **5,473.87 seconds** for reverse retrieval;
this run did not repeat that scan. No training runtime or production runtime is
claimed for this validation-only diagnostic.

- Tracked report: `artifacts/reports/person2_reverse_dev1000_validation_only_v1.json`.
- Local per-pair scores and original report:
  `artifacts/experiments/person2_reverse_dev1000_validation_only_v1/`.
- Frozen models remain in `artifacts/experiments/person2_forward_reproduction_v1/`.
  Their hashes are unchanged from the forward reproduction report.
- XGBoost 3.2.0; lexical feature version `matching-features-v1`.
  All versions and implementation hashes are recorded in the report.
- **75 tests passed in 7.47 seconds**, including explicit failures for training or
  threshold selection in validation-only mode, wrong subset/manifest hashes,
  out-of-scope owners, and attempts to treat the subset as a full-split result.

## Reproduce

Use the environment from `person2_model_evaluation.md`, with a fresh output path.
The directory passed to `--unions` contains the three uploaded union JSONL files
and their producer `report.json`; `--manifest` is the standalone retrieval manifest.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DYLD_LIBRARY_PATH="$PWD/.venv-model/lib/python3.14/site-packages/sklearn/.dylibs"
INPUT_DIR='/Users/parthbhat/Downloads'
.venv-model/bin/python scripts/28_reverse_model_evaluation.py validation-only \
  --sample "$INPUT_DIR/sample_s1.jsonl" --pairs "$INPUT_DIR/pairs.jsonl" \
  --split "$INPUT_DIR/split.json" --allowed "$INPUT_DIR/candidate_pairs_internal.tsv" \
  --baseline artifacts/experiments/person2_forward_reproduction_v1 \
  --unions "$INPUT_DIR" --manifest "$INPUT_DIR/manifest.json" \
  --output artifacts/experiments/person2_reverse_dev1000_validation_only_v1

.venv-model/bin/python -m pytest -q
```

The earlier `compare` command remains the full-split retraining experiment and
requires full training/calibration/development unions. Do not pass this subset to
it or change its provenance checks to accept the subset.
