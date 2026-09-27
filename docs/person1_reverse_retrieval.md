# Person 1: reverse retrieval and frozen candidate unions

This branch adds label-independent target-to-S1 retrieval, preserving the existing
forward candidates. It implements the retrieval part of the revised three-person
plan. Production models and thresholds are unchanged. This is an independently
implemented, bounded sparse experiment, not a reproduction of Ayan's complete
five-view, learned-normalization, stacked-model pipeline.

## Contract

- The index includes **every S1 record**, not just labeled/sample S1s. IDF uses
  unlabeled full-S1 text. There is no country filter or ground-truth candidate injection.
- Score = 0.5 × name character-trigram TF-IDF cosine + 0.5 × address word TF-IDF
  cosine. Each vectorizer uses min_df=2, max_df=0.05 and sublinear TF. Existing
  normalization is unchanged. Missing fields contribute zero; concatenated vectors
  are not renormalized. The version and settings are recorded in the index manifest.
- Retrieve competition ranks 1/3/8 with a strict score floor of 0.35. Rank is one plus
  the number of strictly higher scores. All boundary ties survive: K is a rank
  budget, not a hard edge cap. Top-1/3/8 therefore form nested pools.
- A sample limits **emitted S1 IDs only**. A target with no sampled score above the
  floor can be skipped. A witness subset can also rule out a target only if K witness
  owners score strictly above its best sampled owner. Survivors are scored against
  the full index. Forward targets always bypass these shortcuts for complete
  top-K competitor context. Tests compare shortcuts with ungated retrieval.
- The forward TSV is the finalized pool, not the larger raw retrieval cache. The
  union preserves every forward edge and its original blocking score/rank, adds
  reverse-only edges, and never re-applies a forward top-K cap after union.
- Labels are used only by `27_reverse_union.py` after candidate generation, for
  recall, recovered/lost true links and oracle macro F0.5. Oracle F0.5 is a ceiling,
  not a classifier score. Historical train/calibration/development sets are exploratory.

## Setup and execution

Use a compatible Python environment (the repository pins are intended for an
older Python than the local 3.14 runtime). Install `requirements-retrieval.txt`.
The measured environment versions are recorded in the benchmark report; no claim
is made that the repository's pinned environment was exercised here.

Run from the repository root. Set paths to your existing data and frozen caches:

```bash
python scripts/26_reverse_retrieval.py build \
  --s1 /path/to/train_source1.tsv \
  --output artifacts/experiments/reverse_index --memory-gib 5 --budget-seconds 900

python scripts/26_reverse_retrieval.py retrieve \
  --index artifacts/experiments/reverse_index \
  --s2 /path/to/train_source2.tsv --s3 /path/to/train_source3.tsv \
  --sample /path/to/sample_s1.jsonl \
  --forward-pairs /path/to/candidate_pairs_internal.tsv \
  --k 8 --floor 0.35 --threads 2 --batch-size 1024 --witness-stride 32 \
  --output artifacts/experiments/reverse_full \
  --memory-gib 4 --budget-seconds 43200

python scripts/27_reverse_union.py \
  --sample /path/to/sample_s1.jsonl --split /path/to/split.json \
  --forward-pairs /path/to/candidate_pairs_internal.tsv \
  --forward-cache /path/to/pairs.jsonl \
  --reverse-dir artifacts/experiments/reverse_full \
  --ks 1 3 8 --output artifacts/experiments/reverse_unions
```

For an interrupted retrieval, repeat its exact command with `--resume`. Input and
implementation hashes must match. Recovery truncates uncommitted output to the
last saved batch. The checkpoint includes both output offsets and source row
counts; completed input is skipped on resume. Time budgets count accumulated
active runtime, excluding downtime. Memory gates inspect RSS between batches;
they are observations, not an OS-level memory limit. Output directories must be
new unless explicitly resuming. Keep source inputs immutable throughout a run.

`--max-targets-per-source 10000` is a throughput benchmark only. Such an output
is explicitly marked incomplete, and the union evaluator refuses it. It cannot
establish candidate recall because most of the target corpus was not searched.
`--witness-stride 0` disables the shortcut for correctness/performance comparisons.

Index/joblib artifacts are local trusted artifacts; never load arbitrary downloaded
joblib files. Hash checking detects accidental modifications, not malicious inputs.

## Handoff to Person 2

Completed retrieval writes:

- `reverse_pairs.jsonl`: raw target fields, IDs, source, route, reverse score/rank,
  best/second owner scores, gap to best and margin to best other owner.
- `target_context.jsonl`: global best S1 and top-K positive-score owners for every
  target needed by the finalized forward pool or an emitted reverse edge.
- `manifest.json`: complete marker, full input/index/output hashes, settings,
  counts, timings, observed RSS and implementation hashes.

Completed evaluation writes `union_k1.jsonl`, `union_k3.jsonl`, `union_k8.jsonl`
and `report.json`. Each union row has:

| Field | Meaning |
|---|---|
| `source1_entity_id`, `target_entity_id`, `target_source` | Existing pair identity |
| `target_name`, `target_address`, `target_country` | Raw target text |
| `retrieval_routes` | Forward, reverse, or both |
| `blocking_score`, `blocking_rank` | Original forward metadata; null for reverse-only |
| `forward_score`, `forward_rank` | Explicit aliases for forward provenance |
| `reverse_score`, `reverse_rank` | Full-population sparse score/rank if retained in context |
| `reverse_rank_censored` | Owner absent from retained context; score/rank unknown |
| `reverse_best_s1_id` | Best global owner, with deterministic row-order tie breaking |
| `reverse_best_owner_score`, `reverse_second_owner_score` | Best two owner scores |
| `reverse_gap_to_best`, `reverse_margin_to_best_other` | Pair-versus-competitor evidence, when known |

A forward-only row can still have a reverse rank (e.g. rank 3 in the top-1 union,
or a positive score below the route floor). Route membership is explicitly in
`retrieval_routes`; do not infer it from a non-null rank. Null rank is **not** 0 or
K+1, and missing pair score is **not** 0. Context is a bounded top-K list, not the
complete graph: do not use its length as the total number of competing S1s.

Use the same sample JSONL as the S1 text lookup for the existing feature extractor.
The original 47-feature extractor does not consume blocking-score/rank metadata.
Person 2 must retrain and calibrate after adding candidates/competition features;
passing the existing model over a changed pool is only a diagnostic comparison.
Person 3 can add training-only aliases as a separately versioned retrieval view.

## Current validation status

The full test suite passed with 56 tests, including lossless unions, censored
metadata, boundary ties, full-population competitors, witness-pruning parity,
interrupted resume, changed-input rejection and partial-scan rejection.
The full 2,206,821-row S1 index built in about 40 seconds. A 20,000-target prefix
benchmark produced identical edge hashes with and without witness pruning,
reducing two-thread elapsed time from about 55 to 49 seconds. Those runs measured
about 1.4–1.5 GiB RSS at batch boundaries. Index-build stage observations were about
1.7 GiB; instantaneous peak memory may be higher.

See `artifacts/reports/person1_reverse_benchmark.json` for exact measurements.
The full-corpus retrieval/evaluation is a separate long-running job. Until its
complete `report.json` exists, there is no measured reverse-retrieval recall gain,
no selected K, and no new model/leaderboard score. Use the existing model as control.
