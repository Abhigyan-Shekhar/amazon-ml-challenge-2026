# Person 1: matching-feature handoff

This branch implements a shared, versioned feature extractor, an eight-configuration
XGBoost ablation runner, and batch inference with serialized model/extractor
integrity checks. It does not change retrieval, production normalization,
production thresholds, existing checkpoints, or submission packaging.

## Findings and limitations

On the original **3,000 train / 1,000 calibration / 1,000 development S1** split,
using exactly the frozen capped10/K=20-per-source candidate pool:

| Added families | Features | Calibration F0.5 | Development F0.5 |
|---|---:|---:|---:|
| None (reproduced baseline) | 16 | 0.886242 | 0.889840 |
| Name | 26 | 0.891219 | 0.898957 |
| Address | 29 | 0.892261 | 0.890884 |
| Alternate views | 24 | 0.892695 | 0.899863 |
| Name + address | 39 | 0.898089 | 0.907122 |
| Name + views | 34 | 0.899014 | 0.904922 |
| Address + views | 37 | 0.898250 | 0.901797 |
| All three | 47 | 0.906297 | **0.909947** |

All runs use 200 XGBoost trees, depth 4, learning rate 0.08, regularization 1,
CPU histogram training, seed 2026 and two threads. Every threshold is selected
on calibration S1s only. No pair-level split or internal random holdout is used.
All combinations were enumerated, not generated adaptively from development results.

The all-family model uses threshold **0.60**. Development precision/recall change
from **0.966655 / 0.787757** to **0.971245 / 0.821224**. India changes from
**0.859266 to 0.886578**; US from **0.910564 to 0.925787**. Singleton F0.5 stays
at **0.923077**. Its paired development gain is **0.020107**, with exploratory
95% bootstrap interval **[0.012622, 0.027676]**, resampling 1,000 S1s 2,000 times.

These are **historical-development exploratory results**, not fresh confirmation
or leaderboard performance. In particular, do not compare 0.909947 directly with
the 10k-trained model's 0.889263 on a different 2,000-S1 sample. The local 10k archive
contains models and split IDs, not its candidate training cache; no 10k-training
feature ablation is claimed here. Person 2 must repeat the winning configurations
on the new shared splits and locked evaluation before promotion. Address alone
has an inconclusive gain and slightly lower India/singleton scores.

The compact machine-readable results, input hashes, versions, source-file hashes,
all feature schemas, slices, thresholds and speed measurements are in
`artifacts/reports/person1_feature_ablation.json`. Checkpoints and fitted vocabulary
files stay in ignored `artifacts/experiments/person1_features_final/`; they are
reproducible with the commands below and are not uploaded to Git.

## Features

- **Name:** training-S1 IDF agreement and containment, distinctive disagreement,
  trailing legal-form removal, core exact/Jaccard/edit agreement, initials,
  acronyms and symmetric IDF-weighted fuzzy token alignment. Soft alignment
  uses at most 32 lexicographically ordered unique tokens on each side.
- **Address:** separate postcode, leading building number and explicitly labeled
  unit agreement/conflict/known flags; address-word agreement and an abbreviation
  view. Leading zeros are normalized for building/unit numbers, but not postcodes.
  Explicit ranges/slash numbers and ambiguous components remain unknown.
- **Views:** character-by-character romanization using pinned AnyAscii 0.3.3,
  mixed-alphanumeric OCR substitutions (0/o, 1/l, 5/s), compact-name comparison,
  and script mismatch. Transliteration reads raw NFKC text before the legacy
  punctuation normalizer can remove Indic combining vowel signs.

The first 16 columns exactly reproduce missing-address-v2 features. Missing
address comparisons remain NaN. Optional families have a fixed canonical order:
name, address, views. Training and inference call the same extractor. Neither
labels nor target records are used to fit document frequencies; only unique
training S1 rows are used. transform() never updates the fitted vocabulary.

The address parser is deliberately heuristic. US postcodes require a trailing
state-plus-ZIP pattern; India uses an unambiguous six-digit PIN; France uses a
five-digit code followed by a locality word. A failed parse is unknown, not a
mismatch. Numeric-first street names and unusual international formats remain
limitations. Abbreviation expansion is an alternate US/India view: ambiguous
`st` and `dr` expansions do not replace original text.

AnyAscii is a deterministic transliteration package, not an embedding or an
external business-record lookup. Its character mappings can be imperfect or
empty for unsupported characters. Source and ISC license:
https://github.com/anyascii/anyascii and https://pypi.org/project/anyascii/0.3.3/.
Its use here is experimental; this report makes no new competition eligibility
claim. The production `DIACRITIC_FOLDING_ENABLED` flag remains False.

## Reproduce

Install in a Python environment compatible with the project's pinned base
requirements (the actual measured environment versions are recorded in the report):

```sh
python -m pip install -r requirements-features.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

On macOS, XGBoost also requires an available OpenMP runtime. The measured run used
an already installed scikit-learn OpenMP library through `DYLD_LIBRARY_PATH`;
no system library was installed or modified.

Generate the original candidate cache/split using the existing README workflow,
or use an existing copy. Set `CACHE_ROOT` to the checkout holding those ignored
artifacts. Run from the feature branch checkout:

```sh
CACHE_ROOT='/path/to/existing/checkout'
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/24_feature_ablation.py \
  --sample "$CACHE_ROOT/artifacts/candidates/address_capped10_v4/sample_s1.jsonl" \
  --pairs "$CACHE_ROOT/artifacts/candidates/address_capped10_v4/pairs.jsonl" \
  --split "$CACHE_ROOT/artifacts/experiments/capped10_k20/split.json" \
  --allowed "$CACHE_ROOT/artifacts/experiments/capped10_k20/candidate_pairs_internal.tsv" \
  --output artifacts/experiments/person1_features_final
```

Output must not already exist. `--allowed` is essential for the historical raw
cache: it selects exactly the finalized K=20 pool. Missing/duplicate pairs or
non-disjoint/incomplete splits fail validation. The script accepts the shared
Person 2 split once it has `train`, `calibration`, `validation` lists that exactly
cover the supplied sample. Keep locked evaluation rows outside that sample.

## Integration API

```python
from src.features.matching import PairFeatureExtractor

extractor = PairFeatureExtractor(('name', 'address', 'views')).fit(training_s1_rows)
x_train = extractor.transform(candidate_batch, s1_lookup)
extractor.save('extractor.json')

# Inference: load; do not fit again on test entities.
extractor = PairFeatureExtractor.load('extractor.json')
x_test = extractor.transform(test_candidate_batch, test_s1_lookup)
```

S1 rows use `entity_id`, `business_name`, `business_address`, `country`; candidates
use `source1_entity_id`, `target_entity_id`, `target_source`, `target_name`,
`target_address`, `target_country`. Text values must be strings. Other metadata
is ignored. Keep **raw** text for alternate views. The output is float32, retains
input order, and supports empty batches. Bound batch size in the caller.

Every ablation saves `model.json`, `extractor.json`, `bundle.json`, and its threshold
sweep. Bundle metadata records feature order, versions, training-ID and input
hashes, threshold and model/extractor hashes. Load native XGBoost JSON rather than
an untrusted pickle. The dedicated scorer uses those same files:

```sh
python scripts/25_predict_features.py \
  --bundle artifacts/experiments/person1_features_final/name+address+views \
  --sample /path/to/s1_records.jsonl \
  --pairs /path/to/finalized_candidate_pairs.jsonl \
  --output /path/to/new_scores.tsv --batch-size 4096
```

This produces pair scores/selection flags, not an official submission. The input
must already be the finalized candidate pool from the same retrieval policy;
the scorer does not retrieve or impose top-K. Person 2 integrates that pool with
Person 3's retriever and the official output writer. Do not pass an experimental
47-column model to the existing legacy 16-column production runner.

## Runtime and next step

Three repeated warm-cache 10,000-pair transform timings are recorded per family.
These exclude retrieval and I/O. Full features cost about 2.5x the baseline
feature extraction; name+address offers a useful cheaper alternative. No full
44.9-million-pair runtime or memory gate has been passed for this implementation.
Bounded caches hold at most 32,768 address parses and 32,768 romanized strings.
The ablation runner retains the sample feature matrix in memory; full inference
uses caller-sized batches, although the standalone scorer loads the S1 lookup.

Person 2 should carry forward **name+address** and **all three**, retrain on the
shared larger dataset, recalibrate there, and perform fresh confirmation plus
end-to-end runtime/memory measurement. Person 3 can change candidate generation
independently without editing the feature implementation. Frozen production
packages remain unchanged.
