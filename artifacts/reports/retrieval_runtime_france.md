# Teammate 2 — Retrieval Coverage, Runtime, and France Domain-Shift Diagnostics

Date: 2026-09-26

Branch context: `codex/missing-address-v2`

This report records the requested Teammate 2 investigation. No production model, retrieval configuration, or threshold was changed as part of these diagnostics.

## 1. Retrieval coverage

### Setup

- Frozen 5,000-S1 sample
- Structured name + address blocking
- `max_key_frequency=10`
- Candidate cap consistent with the existing pipeline
- Labels used only for evaluation, not for candidate generation

### Raw candidate recall

- Labeled true pairs: **17,487**
- Retrieved true pairs: **16,202**
- Missed true pairs: **1,285**
- Raw true-pair recall: **92.65%**
- S1 entities with at least one missed true match: **953 / 5,000**

### Misses by country

- India: **819**
- US: **466**

### Script / name-form analysis

- Indic-script target-name misses: **426 / 1,285 = 33.15%**
- Indic-script misses among Indian misses: **426 / 819 = 52.01%**
- Latin-diacritic target-name misses: **102**

Among the **859 non-Indic misses**, normalized name-similarity buckets were:

- `>= 0.90`: **267**
- `0.70–0.90`: **364**
- `0.40–0.70`: **178**
- `< 0.40`: **50**

The low-similarity tail contains cases that look ambiguous/noisy in addition to genuine lexical gaps, so those cases should not all be interpreted as clean retrieval failures.

### Latin-diacritic normalization ablation

Isolated change: strip Latin diacritics while keeping the rest of the retrieval setup fixed.

- Candidate pairs: **608,840 -> 619,437** (`+1.74%`)
- Raw true-pair recall: **92.65% -> 93.06%**
- True pairs recovered: **72**
- Previously retrieved true pairs lost: **0**
- Latin-diacritic misses recovered: **71 / 102 = 69.61%**
- Indic-script misses recovered: **1 / 426 = 0.23%**

Interpretation: Latin-diacritic normalization is a clean but limited retrieval improvement. It does not solve the main cross-script problem.

### Indic transliteration diagnostic

A straightforward Indic-to-Latin transliteration diagnostic was run on the 426 Indic-script misses.

- Cases gaining at least one exact S1/target token overlap after transliteration: **112 / 426 = 26.29%**
- Cases still with zero exact-token overlap: **314 / 426 = 73.71%**

Interpretation: simple transliteration alone is insufficient for the current exact-token blocker. Broader phonetic, fuzzy, or multilingual retrieval would be needed to address most cross-script misses.

## 2. Runtime profiling

### Historical full-run comparison

- v1 runtime: **2165.159 s**
- v2 runtime: **3423.709 s**
- Runtime increase: **+58.13%**
- Candidate count v1: **44,931,896**
- Candidate count v2: **44,931,896**
- Candidate delta: **0**
- Peak retrieval RSS v1: **7.55 GiB**
- Peak retrieval RSS v2: **4.91 GiB**

Because the candidate count is identical, the regression is not explained by retrieving more candidates.

### Classifier-only benchmark

Controlled benchmark on identical first 100,000 saved validation pairs:

- v1 `predict_proba`: approximately **0.073–0.085 s**
- v2 `predict_proba`: approximately **0.072–0.074 s**

There is no meaningful classifier-inference slowdown.

### Feature-construction benchmark

Same-machine benchmark on identical 200,000 candidate pairs:

- v1 feature construction: **9.844 s**
- v2 feature construction: **10.470 s**
- v1 throughput: **20,317.6 pairs/s**
- v2 throughput: **19,101.3 pairs/s**
- v2 feature-construction slowdown: **6.37%**

This is far below the historical **58.13%** full-run regression, so the neutral-missing-address feature/model change does not explain most of the runtime increase.

A full Colab profiling attempt also showed strong machine/environment dependence: retrieval alone was projecting roughly 58 minutes, so cross-machine wall-clock totals should not be treated as directly comparable.

### Runtime conclusion

The measured evidence localizes the regression away from classifier inference and shows only a modest feature-construction cost increase. Most of the historical full-run delta is therefore outside the classifier itself and is consistent with retrieval / I/O / environment / run-to-run system effects rather than the missing-address model change alone.

## 3. France domain-shift diagnostics

France is test-only, so no France accuracy, recall, precision, or F0.5 is reported.

### Test Source1 population

- France: **259,452 S1**
- India: **809,986 S1**
- US: **663,106 S1**
- Missing Source1 address rate: **0%** for all three countries

### Diagnostic setup

- Deterministic sample: **1,000 S1 each** from France, India, and US
- Same frozen v2 model
- Threshold: **0.585**
- Same structured retrieval setup
- `max_key_frequency=10`
- `K=20` per source
- Address-path enabled

Retrieval over the 3,000-S1 diagnostic sample scanned **9,969,589** target rows and produced **321,645** scored retrieval pairs.

### Country comparison

| Metric | France | India | US |
|---|---:|---:|---:|
| S1 sample | 1,000 | 1,000 | 1,000 |
| Avg candidates / S1 | **28.491** | 27.384 | 23.983 |
| Median candidates / S1 | **34.5** | 34.0 | 24.0 |
| Zero-candidate rate | **0.0%** | 0.3% | 0.0% |
| Avg predicted matches / S1 | **3.871** | 2.884 | 2.905 |
| Zero-predicted-match rate | **4.4%** | 8.3% | 7.5% |
| Mean candidate score | **0.1599** | 0.1196 | 0.1374 |
| Median candidate score | **0.00416** | 0.00086 | 0.00046 |
| P90 candidate score | **0.9550** | 0.6676 | 0.9377 |
| Target missing-address rate | 0.944% | 0.738% | 1.159% |
| Non-ASCII target-name rate | **22.246%** | 11.949% | 4.812% |

### France conclusion

France does **not** show candidate starvation. Instead, under the frozen model it has a shifted candidate and score distribution, including:

- more candidates per S1 than India/US,
- more predicted matches per S1,
- fewer S1 entities with zero predicted links,
- a higher upper-tail candidate-score distribution,
- substantially more non-ASCII target names.

Because France has no ground-truth labels, these observations are evidence of **domain shift only**. They do not establish whether France precision, recall, or F0.5 is better or worse.

## 4. Constraints / production status

- No production checkpoint changed.
- No production threshold changed.
- No production retrieval configuration changed.
- The v2 threshold remains **0.585**.
- Recorded v2 package SHA-256 remains:

`10f28301c9f5fcca26ae01b714211df9b129eeeb376691e48769770653e9f4f6`
