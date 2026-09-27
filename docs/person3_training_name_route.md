# Person 3: training-only name alias retrieval

## Route and safeguards

`training_name_alias` is a separate target-to-S1 exact alias lookup, implemented
in `src/blocking/training_name.py`. It keeps the repository's normalization and
all original name text unchanged. It learns whole-name equivalences only from
the `train` S1 IDs in the frozen split file and their labeled training pairs.
An equivalence requires at least two distinct linked S1 IDs and two distinct
target IDs. The map records every supporting training pair and ID, the complete
input-file SHA-256 values, a hash of the training partition IDs, and a content
hash for the versioned map. A single linked pair cannot create a map.

This captures recurring observed transliterations, abbreviations, and spelling
variants when those surface forms recur in distinct training entities. It does
not use a transliteration service, business lookup, or external alias list, and
does not invent unseen spellings. Names with no learned alias also pass through
the route as exact normalized-name keys. That gives it ordinary exact-name
blocking behavior alongside the learned variants.

Candidate generation reads no labels. It emits route membership, alias support,
score, and rank. The evaluation union retains every forward candidate and its
original blocking score/rank, and adds route metadata on overlap or reverse-only
candidate rows. Person 1's code and reverse fields are unchanged.

## Reproduction

Run from the repository root with the same split/sample and frozen forward
artifacts used by Person 1. The split JSON must include disjoint `train`,
`calibration`, and `development` S1 ID lists. Paths below are examples.

```bash
python scripts/28_training_name_route.py build \
  --s1 /path/to/train_source1.tsv --s2 /path/to/train_source2.tsv \
  --s3 /path/to/train_source3.tsv --labels /path/to/train_ground_truth.tsv \
  --split /path/to/split.json --min-support 2 \
  --output artifacts/experiments/name_alias/map.json

python scripts/28_training_name_route.py retrieve \
  --s1 /path/to/train_source1.tsv --s2 /path/to/train_source2.tsv \
  --s3 /path/to/train_source3.tsv --sample /path/to/sample_s1.jsonl \
  --map artifacts/experiments/name_alias/map.json \
  --output artifacts/experiments/name_alias/route

python scripts/28_training_name_route.py evaluate \
  --sample /path/to/sample_s1.jsonl --split /path/to/split.json \
  --labels /path/to/train_ground_truth.tsv \
  --forward-pairs /path/to/finalized_forward_pool.tsv \
  --forward-cache /path/to/raw_forward_pairs.jsonl \
  --name-route artifacts/experiments/name_alias/route \
  --output artifacts/experiments/name_alias/ablation
```

Only the `train` partition labels create mappings. Calibration and development
labels are loaded only by the final evaluation command, after the route output
exists, to measure candidate recall. No test labels are read. Treat development
and calibration metrics as exploratory; this is a retrieval ablation, not a
model-score claim.

## Ablation status

The dataset, split IDs, and frozen candidate caches are not present in this
checkout, so the existing train/calibration/development split metrics and real
corpus runtime could not be measured here. `tests/test_training_name_route.py`
contains synthetic cases for independent-entity support, training-only map
construction, provenance, raw-name preservation, and candidate metadata. Run the
commands above to produce `report.json`, which reports route-only and forward
candidate recall, union recall, new/lost true links, added candidates per S1,
and route/evaluation runtime. The report does not estimate reranker or end-to-end
model scores.

## Person 2 handoff

Build and freeze the alias map from training IDs only; add `name_alias_pairs.jsonl`
to the candidate pool with a lossless union. Keep `retrieval_routes`, forward
`blocking_score`/`blocking_rank`, and `name_alias_score`/`name_alias_rank` as
separate provenance. Retrain and recalibrate after the pool changes. The current
branch has no measured corpus recall gain and makes no end-to-end score claim.
