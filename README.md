# Amazon ML Challenge 2026 — business entity resolution

The real datasets have been audited. The archived fuzzy-feature tree v1 scores **0.890226 macro F0.5** on the development holdout and **0.875554 on a fresh, disjoint 2,000-S1 confirmation set**. The corrected missing-address v2 tree scores **0.892146** and **0.877411** on those same splits, using the calibration-selected threshold **0.585**. Its full-test inference is complete, officially validated with zero errors and warnings, and packaged separately. A separate exact-match fallback (**0.690681** locally) is also packaged and officially validated. **Do not attribute the stronger model’s scores to the fallback files.** No leaderboard submission has been made.

This repository contains the completed audit, experiments, reproducible scripts, reports, tests, official validator integration, and packaging workflows. Large outputs, checkpoints, raw data, and the prepared GPU validation package remain local and excluded from Git; teammates must reproduce them or arrange a private artifact transfer.

Archived v1 full-test package: `outputs/fuzzy_tree_v1_submission.zip` (**291.8 MB**, after final documentation refresh and recompression). Both TSVs contain **1,732,544 rows**. The model scored **44,931,896 candidates**, selected **5,196,796 links**, and left **131,317** S1 match lists empty. Runtime: **2,165.16 seconds (36.1 minutes)**; peak measured retrieval RSS: **7.55 GiB**. See [official validation](artifacts/reports/fuzzy_official_validation.json) and [runtime](artifacts/reports/fuzzy_runtime.json). The test partition has **9,969,589 targets**; 10,320,219 is the training target count.

Selected local v2 package: `outputs/fuzzy_missing_address_v2_submission.zip`. Official validation checked all 1,732,544 S1 rows and all target IDs with **zero errors and zero warnings**. The ZIP follows the challenge layout: `output/` contains the two TSVs, `Documentation_template.md` is at the archive root, and `code/business_entity_resolution/` contains the runnable code, pinned requirements, example config, checkpoint, and run manifests. The source template is [Documentation_template.md](Documentation_template.md). **Explicit human approval is required before any leaderboard upload.**

### Missing-address v2 promotion

The v1 feature code assigned zero address similarity when either address was absent, which made missing data resemble a confirmed mismatch. V2 keeps the same 16-feature HistGradientBoosting design and changes unavailable address comparisons to `NaN`: address token overlap, exactness, numeric overlap/conflict, length ratio, and three RapidFuzz similarities. The existing address-missing flag remains. The v1 checkpoint and package are preserved; v2 has a new checkpoint under `artifacts/experiments/capped10_missing_address_v2/`.

V2 uses the original train/calibration/dev split and the disjoint confirmation set. Its calibration-only threshold search selected **0.585**. Reusing the v1 numeric threshold **0.64** on the new score scale gives dev macro F0.5 **0.889913**, below v1, so the v2 production threshold is 0.585. At the fixed 0.64 diagnostic cutoff, 17 of the original 515 missed true pairs now cross it. Dev precision/recall are **0.962720/0.797769**; confirmation precision/recall are **0.952112/0.792469**. See [v2 ablation](artifacts/reports/missing_address_v2.json).

Full-test v2 inference scored **44,931,896 candidates** and selected **5,323,479 links**, a net increase of **126,683** (2.4%) over v1. It added 163,743 links and removed 37,060, changing 176,821 S1 match lists; empty lists fell from 131,317 to 124,102. Runtime was **3,423.71 seconds (57.1 minutes)** versus 2,165.16 seconds for v1, a **58.1% increase**; peak measured retrieval RSS was **4.91 GiB** versus 7.55 GiB. The run stayed within the 4,200-second and 10-GiB gates, but the runtime cost is material. These are unlabeled test predictions, not a test accuracy measurement.

A controlled 100,000-pair `predict_proba` benchmark found about **1.38 million pairs/second** for both classifiers. V2's full-run retrieval phase was also slower (1,697 versus 1,119 seconds), despite unchanged retrieval code and candidate count. The single full-run timing does not isolate a NaN-specific cost; it is still the measured end-to-end runtime to budget for. See [runtime benchmark](artifacts/reports/missing_address_runtime_benchmark.json).

Manual review of 30 missed true pairs and 23 false positives is recorded in `artifacts/validation/hgb_score_diagnostic/stratified_manual_review.csv` and reproducibly annotated by `scripts/23_annotate_review.py`. All 16 missed-pair `other` cases now have a subtype and exclusion reason. Four transliterations plus two plausible semantic aliases make **6/30 (20%)**, below the 40% embedding gate; the broader name-variation signal is **11/30 (36.7%)**. Seven `other` labels remain unclear, so the conservative upper bound is **13/30 (43.3%)** if every unclear case were a valid semantic alias. The embedding gate is therefore **indeterminate**, and no embedding feature is promoted on this evidence. Label noise plus shared-address ambiguity is **4/30 (13.3%)**, below the 50% ceiling gate. See [gate report](artifacts/reports/annotation_gate_v2.json).

Before any leaderboard upload, freeze the final checkpoint and threshold, complete full-test inference and official ID/format validation, package the outputs and runnable code, and obtain explicit human approval. The local package is not an upload authorization.

## Completed

- [x] Audited all seven supplied TSVs, with input SHA-256 hashes and the official validator retained in `utils/`.
- [x] Confirmed schema, missingness, ID uniqueness, label coverage, 0/1/many distribution, and S2/S3 ownership.
- [x] Implemented and tested exact per-S1 macro F0.5, including singleton behavior.
- [x] Measured local compute and global character-retrieval feasibility before scaling.
- [x] Retrieved candidates for 5,000 uniformly sampled S1s against **all 10,320,219 training targets**.
- [x] Kept 3,000 S1s for classifier training, 1,000 for threshold calibration, and 1,000 for held-out reporting. No random pair split.
- [x] Measured name blocking, added address blocking after a recall bottleneck, and compared cosine, logistic features, and a shallow boosted-tree matcher.
- [x] Ran country-transfer sanity checks, singleton analysis, and a fresh 2,000-S1 confirmation; retained separate model checkpoints.
- [x] Measured common-key pruning and fast edit features to make the stronger model feasible on CPU. The final candidate pool is capped at 20/source and excludes keys shared by more than 10 S1s.
- [x] Adapted output to official comma-separated lists, including one candidate row per S1.
- [x] Prepared a Kaggle/Colab reranker validation package with **227,362 fixed candidate pairs**, compliance checks, and runtime gates.
- [x] Passed **21 tests**, including a missing-address encoding regression test.

Full-test package: `outputs/exact_core_v1_submission.zip` (**255 MB**). Official validator passed with **zero errors and zero warnings**, including ID checks. Both TSVs contain **1,732,544 rows**; the matcher scored **37,420,131 candidates** and selected **3,879,980 links**. Runtime: **822.94 seconds**, peak measured retrieval RSS: **4.73 GiB**.

## Dataset findings

| Partition | S1 | S2 | S3 | Countries |
|---|---:|---:|---:|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | India, US |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | India, US, France |

Training S1 distribution: **123,247 singletons (5.58%)**, **119,157 one-match entities (5.40%)**, and **1,964,417 multi-match entities (89.02%)**. Labels contain 7,638,365 links. Every target in the ground truth belongs to exactly one S1; no global assignment constraint has been applied yet.

No duplicate IDs, malformed records, missing/unknown S1 labels, duplicate truth matches, or invalid target references were found. Names and countries are populated. Missing addresses: train S2 **168,967**, train S3 **175,916**, test S2 **129,408**, test S3 **136,098**. No France validation is claimed.

See [full audit](artifacts/reports/dataset_audit.json) and [input hashes](artifacts/reports/input_manifest.json). Raw data remains outside Git; `config.json` points to the supplied Downloads paths.

## Measured experiments

All scores below use the same 1,000 held-out S1s, including 52 singletons. Thresholds and model retention use the separate calibration groups. Precision and recall are diagnostic micro link metrics. K is the **effective recorded cap**, per target source.

| Retrieval / matcher | K | Candidate link recall | Macro F0.5 |
|---|---:|---:|---:|
| Name/structured blocks → character cosine | 50 | 0.7580 | 0.6910 |
| Same candidates → logistic lexical features | 50 | 0.7580 | 0.7719 |
| Add address blocks → character cosine | 100 | 0.9379 | 0.7168 |
| Add address blocks → logistic lexical features | 100 | 0.9379 | 0.8219 |
| Add address blocks → shallow boosted trees | 100 | 0.9379 | **0.8744** |
| Cap key frequency at 100 + 20/source → token-feature tree | 20 | 0.9062 | 0.8277 |
| Cap key frequency at 10 + 20/source → token-feature tree | 20 | 0.9379 | 0.8438 |
| Same pool + 6 RapidFuzz edit features → tree | 20 | 0.9379 | **0.8902** |
| Exact name-core/address fallback → token-only logistic | 100 | 0.5332 | 0.6907 |

The archived v1 fuzzy-feature tree has **0.9654 precision**, **0.7906 recall**, and **0.9231 singleton F0.5** on the development holdout. It uses scikit-learn HistGradientBoosting with 200 iterations, depth 4, and no internal random pair validation. Its inputs are ten token/numeric/country/length features plus six RapidFuzz edit similarities. It is a CPU model, not a neural reranker.

On a **fresh 2,000-S1 confirmation set** excluded from all prior training, calibration and holdout groups, the unchanged v1 model and threshold score **0.875554 macro F0.5**, **0.9588 precision**, **0.7839 recall**, and **0.8779 singleton F0.5**. Candidate link recall is **0.9223**; all matches are retrieved for **79.45%** of S1s. India and US F0.5 are **0.8359** and **0.9031**. The bootstrap interval is approximately **0.8658–0.8853**, covering sampling uncertainty only, not France/domain-shift risk. See [v1 independent confirmation](artifacts/reports/fuzzy_confirmation.json) and [v2 results](artifacts/reports/missing_address_v2.json).

The earlier full-character-feature tree scored 0.8744 on the development holdout but is more expensive to deploy. Country-transfer checks using country-specific logistic training/calibration scored **India → US 0.8382**, **US → India 0.7267**. The latter is a material warning against assuming equal cross-country performance. No France validation is claimed.

Compact measured artifacts: [results CSV](artifacts/reports/results.csv), [experiment/slice summary](artifacts/reports/experiment_summary.json), [singleton analysis](artifacts/reports/singleton_analysis.json). Full run caches, split IDs, score tables and checkpoints are local under `artifacts/experiments/`.

## Why the architecture changed

Local hardware: 10 ARM CPU cores, 16 GiB RAM, no detected CUDA GPU. Free-tier Kaggle/Colab access selects **Mode B**; allocation is not guaranteed.

A 20,000-record sample projects **14.35 GiB for a global target character matrix alone**, excluding text, vocabulary, temporary products and candidate output. A rough linear query projection is **137 hours**. Therefore the original small-data global TF-IDF runner refuses input above 500 MB.

The scalable sample workflow uses label-independent exact-name, rare-name-token-pair, name-plus-address-clue, and rare-address-token-pair blocks. Country is never a hard filter. Character TF-IDF similarities are calculated within the resulting pool. The initial single-token route was stopped after 39 million comparisons per million targets on only 5,000 queries. The name-only matcher used K=50/source; address and exact runs used K=100/source. Adding address blocks raised full-pool sample recall from 75.3% to 93.0%; the held-out slice recall is 93.8%.

The selected full-test run uses a compact hashed S1 blocking-key index, drops keys with more than ten S1s, and keeps 20 candidates per source by token-overlap ranking before batched fuzzy-tree inference. The entire 16-feature matcher benchmarked at approximately **58,088 pairs/second** on 100,000 representative pairs; this excludes retrieval and file serialization. Name/address token document frequencies are fitted without labels on each S1 partition.

The preserved full-test fallback uses a much cheaper exact sorted name-core OR address-token-set index, followed by the same recorded token-overlap cap and frozen token-feature logistic matcher as its validation run. Legal-form removal affects only this auxiliary key; conservative normalized and raw fields remain intact. This fallback trades recall for runtime and is **not recommended for spending a leaderboard slot while the stronger model is available for further work**.

## Reproduce locally

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/00_compute.py --external-gpu yes --config config.json
python scripts/10_stream_eda.py --root /path/to/resources
```

The resource root contains `train/train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv` and corresponding files under `test/`. Copy `config.example.json` to ignored `config.json` and replace paths. The official truth field is a comma-separated list or blank; a legacy JSON-list adapter remains for old fixtures.

Measured sample pipeline (use fresh output directories; existing checkpoints are never overwritten):

```sh
python scripts/11_sample_candidates.py --root /path/to/resources/train --address-path --output artifacts/candidates/new_address
python scripts/12_sample_experiment.py --candidates artifacts/candidates/new_address --output artifacts/experiments/new_address --k 100
python scripts/13_meta_experiment.py --run artifacts/experiments/new_address --sample artifacts/candidates/new_address/sample_s1.jsonl --output artifacts/experiments/new_tree
```

For portability, the sample is generated by `10_stream_eda.py`. Pass `--sample` to the candidate script when using a nondefault audit directory. Token-frequency cache reuse is guarded by the source-file SHA-256. The current runs all use the hashed supplied inputs.

Full fallback reproduction:

```sh
python scripts/11_sample_candidates.py --root /path/to/resources/train --exact --output artifacts/candidates/new_exact
python scripts/12_sample_experiment.py --candidates artifacts/candidates/new_exact --output artifacts/experiments/new_exact --ranking blocking --k 100
python scripts/15_full_exact_baseline.py --test-dir /path/to/resources/test --model-dir artifacts/experiments/new_exact --output outputs/new_exact --budget-seconds 1800
python utils/validate_submission.py --matching outputs/new_exact/matching_results.tsv --candidate outputs/new_exact/candidate_pairs.tsv --test-dir /path/to/resources/test --check-ids
```

The small-data runner `02_baseline.py` is retained for fixtures and small subsets. Do not point it at the multi-million-record dataset.

## Selected CPU model and full-test run

```sh
python scripts/11_sample_candidates.py --root /path/to/resources/train --address-path --max-key-frequency 10 --output artifacts/candidates/new_capped
python scripts/12_sample_experiment.py --candidates artifacts/candidates/new_capped --output artifacts/experiments/new_capped --ranking blocking --k 20
python scripts/13_meta_experiment.py --run artifacts/experiments/new_capped --sample artifacts/candidates/new_capped/sample_s1.jsonl --output artifacts/experiments/new_token_tree --cheap-only
python scripts/21_fuzzy_experiment.py --run artifacts/experiments/new_capped --candidates artifacts/candidates/new_capped --baseline-tree artifacts/experiments/new_token_tree --output artifacts/experiments/new_fuzzy_tree
PYTHONHASHSEED=2026 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/18_full_structured.py --test-dir /path/to/resources/test --model-dir artifacts/experiments/new_capped --tree-dir artifacts/experiments/new_fuzzy_tree --max-key-frequency 10 --output outputs/new_fuzzy --budget-seconds 2400
python scripts/17_package_baseline.py --run outputs/new_fuzzy --test-dir /path/to/resources/test
```

For the missing-address v2 checkpoint, reuse the **same finalized candidate pool and split** from the commands above, then run:

```sh
python scripts/22_missing_address_ablation.py --run artifacts/experiments/new_capped --candidates artifacts/candidates/new_capped --confirmation artifacts/candidates/confirmation_cap10 --missed artifacts/validation/hgb_score_diagnostic/validation_retrieved_true_pairs_missed_at_064.csv --output artifacts/experiments/new_missing_address_v2
python scripts/20_confirm_frozen.py --candidates artifacts/candidates/confirmation_cap10 --model artifacts/experiments/new_missing_address_v2/model.joblib --original-split artifacts/experiments/new_capped/split.json --output artifacts/validation/confirmation/missing_address_v2.json
PYTHONHASHSEED=2026 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/18_full_structured.py --test-dir /path/to/resources/test --model-dir artifacts/experiments/new_capped --tree-dir artifacts/experiments/new_missing_address_v2 --max-key-frequency 10 --output outputs/new_missing_address_v2 --budget-seconds 4200 --memory-gib 10
python scripts/17_package_baseline.py --run outputs/new_missing_address_v2 --test-dir /path/to/resources/test
```

The packaging command validates both output files with full ID checks before writing the archive. It defaults to `BlackList_submission.zip`; use `--archive-name fuzzy_missing_address_v2_submission.zip` to reproduce the named v2 artifact. The archive contains only the official output folder, the completed documentation template, and the self-contained code folder described above. All Python dependencies in `requirements.txt` are pinned to exact versions.

The `--missed` file only produces the historical 515-pair diagnostic count; it does not affect model fitting or threshold selection. Generate the independent confirmation sample first if those frozen local artifacts are unavailable.

`19_confirmation_sample.py` and `20_confirm_frozen.py` produce and evaluate a disjoint confirmation sample. They never recalibrate the frozen threshold. The current selected local model is `artifacts/experiments/capped10_missing_address_v2/model.joblib`, threshold **0.585**. The archived v1 checkpoint remains `artifacts/experiments/capped10_fuzzy_tree/model.joblib`, threshold **0.64**. Checkpoints, raw data, pair caches, and outputs remain ignored by Git.

The intermediate token-only full-test attempt was stopped before prediction output after the measured fuzzy-feature improvement; its checkpoint and progress remain under `outputs/structured_tree_v1/`. It is not a complete submission. The validated exact fallback remains intact.

## Official output contract

`matching_results.tsv`: `source1_entity_id<TAB>matched_entity_ids`.

`candidate_pairs.tsv`: `source1_entity_id<TAB>candidate_entity_ids`.

Each file contains exactly one row per test S1. Lists contain comma-separated S2-/S3- IDs; singleton lists are blank. Final matches must be contained in the exact finalized candidate pool supplied to the matcher. Our pipeline enforces containment even though the provided validator only warns about it. Run the official validator with `--check-ids` for full membership checks.

## GPU validation package

Local directory: `artifacts/gpu_jobs/reranker_address_v2/`. Upload privately to Kaggle/Colab if competition data rules allow it. It contains the fixed pair texts, scoring script, split metadata and instructions. There are no model weights or external business records in the package.

```sh
pip install torch transformers huggingface_hub sentencepiece
python score_pairs.py --pairs pairs.jsonl --output-dir reranker_pretrained --budget-seconds 3600
```

The script checks repository license/count before weights, pins the resolved revision, records compliance evidence, benchmarks text-length quantiles, and gates full inference using a 50% runtime margin. A partial cache cannot be imported. Download the complete output directory and run locally:

```sh
python scripts/14_gpu_sample_job.py import --job artifacts/gpu_jobs/reranker_address_v2 --cache /path/to/reranker_pretrained --output artifacts/validation/pretrained_reranker.json
```

[Compliance evidence](artifacts/compliance/model_compliance.md) verifies the reranker's repository metadata. BGE-M3 is still deferred because parameter-count evidence is incomplete. **No neural inference or fine-tuning has run.** Reverify any selected neural model before final packaging.

## Remaining work

- [ ] **Team handoff:** Obtain the ignored `outputs/fuzzy_missing_address_v2_submission.zip` through a private artifact transfer and verify SHA-256 `10f28301c9f5fcca26ae01b714211df9b129eeeb376691e48769770653e9f4f6`. The Git branch contains code and aggregate reports, not raw data, checkpoints, or submission TSVs.
- [x] **Resolve the embedding gate:** Seven `label_valid_under_spec=unclear` missed-pair rows in `stratified_manual_review.csv` formally closed as unresolved under agreed protocol (no external records, no unverified embeddings). Embedding gate is indeterminate (observed alias share 6/30 = 20%); no embedding feature promoted. See `artifacts/reports/annotation_gate_v2.json`.
- [ ] **Check transfer and runtime:** France has no supplied training labels; investigate domain shift separately. Recheck v2's 57.1-minute full-run time on the intended machine before planning any time-sensitive inference. The controlled classifier benchmark did not show a NaN-specific prediction slowdown.
- [x] Complete full-file official validation and package the fallback.
- [x] Finish, officially validate, and package the selected fuzzy-tree full-test run (`outputs/fuzzy_tree_v1`).
- [x] Correct missing-address feature encoding, confirm gains on both validation splits, run and officially validate full-test v2, and package `outputs/fuzzy_missing_address_v2_submission.zip`.
- [x] Run and import the GPU pretrained-reranker benchmark. On the shared 1,000-S1 validation split, BGE reranking scored 0.6482 macro F0.5 versus 0.8902 for the selected fuzzy CPU tree; it is not promoted. See [GPU reranker validation](artifacts/reports/pretrained_reranker_validation.json).
- [ ] Improve remaining retrieval misses (especially altered scripts/names) toward 99% recall; dense/BM25/RRF additions require measured gains and feasible runtime.
- [ ] After a successful pretrained baseline, prepare supervised hard negatives and at most one mining round; preserve v1/v2 separately.
- [ ] Consider global consistency only as a measured experiment; the uniqueness audit permits testing it, but does not establish unseen-test behavior.
- [x] Run fresh disjoint confirmation after model selection, without further tuning.
- [x] Test a versioned normalization that preserves Unicode combining marks, particularly for noisy target scripts; never change frozen model preprocessing in place. **Done (2026-09-26): Latin diacritic folding tested end-to-end on capped10 at frozen threshold 0.585. Confirmation gain +0.00028 — REJECTED for production. `DIACRITIC_FOLDING_ENABLED=False` in `src/normalize.py`. Results in `artifacts/validation/diacritic_fold_end_to_end.json`.**
- [ ] Seek explicit human approval before any leaderboard upload. Recheck neural compliance if a future neural model is selected.

For parallel work, start from the latest pushed commit and use separate output directories. Remaining research tasks include retrieval-recall improvements, France/domain-shift investigation (France has no supplied training labels), and review of the final fuzzy full-test artifacts. Do not overwrite the frozen checkpoints or claim a leaderboard score from local validation.

`artifacts/submissions/submission_log.csv` remains empty. No submission slot has been used. Never exceed five uploads per challenge calendar day; the challenge timezone and any additional rules still need confirmation. Reserve the final two hours for packaging and validation.

## Experimental matching-feature handoff

Person 1's versioned extractor and controlled XGBoost ablations are documented in
[the feature handoff](docs/person1_matching_features.md). On the original fixed
3,000/1,000/1,000 split, all three added feature families score **0.909947** versus
**0.889840** for the reproduced 16-feature control. These are exploratory results
with higher feature-extraction cost; fresh confirmation and full-runtime checks
are required before production promotion. Existing production packages are unchanged.

## Reverse-union model evaluation

Person 2's [model-evaluation handoff](docs/person2_model_evaluation.md) reproduces
the frozen forward XGBoost controls exactly and provides a tested runner for
K=1/3/8 reverse unions with target competition features. The measured 47-feature
forward control scores 0.909947 exploratory development macro F0.5; the reverse
comparison awaits completed union artifacts. Production bundles are unchanged.

The completed `reverse_dev1000_v1` **validation-only** diagnostic is documented
[here](docs/person2_validation_only_reverse.md). With unchanged 47-feature models
and calibration threshold, reverse K=1/3/8 score 0.900903/0.898848/0.897475 versus
0.909947 forward. Candidate recall improves, while precision and macro F0.5 fall.
Training and calibration pools are unchanged; this is not a full-split retraining
result. The explicit mode verifies exact subset provenance and validation scope.
