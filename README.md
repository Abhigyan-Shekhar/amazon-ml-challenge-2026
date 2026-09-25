# Amazon ML Challenge 2026 — incremental entity resolution

Status: CPU baseline implemented and tested on synthetic fixtures. **No challenge data, official validator, real validation score, or approved submission is available yet.** No neural model has been downloaded or trained. No external business data is used.

The current machine has 10 CPU cores, 16 GiB RAM, no detected CUDA/MPS device, and approximately 20 GiB free disk. Mode B is selected because free-tier Kaggle/Colab access is available; actual GPU allocation must be checked when starting a job. See `artifacts/compute_report.json`.

## What's done

- [x] Compute detection and Mode B selection: local CPU plus free-tier Kaggle/Colab GPU access.
- [x] Conservative normalization with raw-field preservation and open-set country strings.
- [x] Canonical TSV loaders and EDA reporting, ready for the actual dataset.
- [x] Exact per-S1 macro F0.5 evaluator, singleton handling, and grouped calibration/validation split.
- [x] CPU character TF-IDF candidate retrieval against S2 and S3 independently.
- [x] Threshold search, candidate recall/complete coverage, and validation slice reporting.
- [x] Provisional submission generation with internal ID, duplicate, containment, and round-trip checks.
- [x] Per-run manifests, input hashes, experiment logging, and an initialized submission log.
- [x] Portable GPU pair export, reranker scoring with compliance/runtime gates, and score import/evaluation scripts. **GPU execution is still unverified.**
- [x] Local correctness suite: **15 tests passing**, including synthetic end-to-end EDA and baseline execution. These are not competition performance results.

## What's left

The next required step is to obtain the challenge datasets, official schema/sample submission, rules, and validator.

- [ ] Inspect all real train/test files; adapt the provisional input/output formats to the official contract.
- [ ] Run real-data EDA: missingness, countries, singleton/multi-match prevalence, and shared S2/S3 targets.
- [ ] Benchmark retrieval on a representative sample, then measure the baseline on grouped validation.
- [ ] Run the official submission validator and produce the first valid baseline submission package.
- [ ] Verify neural model license and parameter evidence, execute a sample GPU benchmark, and evaluate the pretrained reranker if feasible.
- [ ] Add BM25/RRF or dense retrieval only if measured candidate recall/coverage warrants it.
- [ ] Implement grouped supervised training and hard-negative fine-tuning only after a successful pretrained baseline; preserve checkpoints and limit initial mining to one round.
- [ ] Evaluate optional pair features, calibration, and global consistency only where data supports them and validation improves.
- [ ] Complete actual country-transfer checks, singleton analysis, measured ablations, and methodology conclusions.
- [ ] Freeze the strongest validated configuration, recheck compliance, implement final selected-model test inference, and package officially validated outputs.
- [ ] Obtain explicit human approval for any leaderboard upload and enforce the five-per-challenge-day budget.

No real validation score, trained neural checkpoint, or final submission is available yet. Fine-tuning and additional retrieval/model branches remain deliberately deferred until measurements justify them.

## Run locally

Use Python 3.11+ in an isolated environment. The tested local runtime was Python 3.14; the exact installed package snapshot is in `artifacts/environment_versions.json`. GPU environments have separate dependencies.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/00_compute.py --external-gpu yes
```

Copy `config.example.json` to `config.json` and supply the real input paths once available. Paths inside the config resolve relative to that config file. Then:

```sh
python scripts/01_eda.py --config config.json
python scripts/02_baseline.py --config config.json --experiment-id char_v1
```

The runner refuses to reuse an experiment directory. Results, immutable per-run manifest, split IDs, calibration sweeps, singleton score analysis, candidate scores, and provisional test TSVs are stored under `artifacts/experiments/char_v1/`. Experiment summaries append to `artifacts/experiments/results.csv`. Do not treat synthetic test results as competition evidence.

## Provisional input/output contract

This contract is an adapter boundary, **not a claim about the official format**. Update it after inspecting the actual files/sample submission. Preserve IDs as strings, including leading zeros.

Each source TSV requires `entity_id`, `business_name`, `business_address`, `country`. Additional columns are preserved. Blank text is allowed. Blank/duplicate IDs and cross-source ID collisions stop the pipeline for investigation. If the official data uses scoped IDs, add explicit source disambiguation before proceeding.

Canonical training labels require exactly one row per S1, with columns `source1_entity_id` and `matched_entity_ids`. The latter is a JSON string list or an empty field. Missing S1 label rows are an error; they are never silently converted to singletons. A positive-links-only official label file will need a documented adapter based on its actual completeness guarantee.

Provisional candidate output has one row per `(source1_entity_id, target_entity_id)`. Matching output has one row per S1, with a JSON list of matched IDs or a blank singleton field. Pandas TSV quoting is used. The official separator/list encoding must be confirmed. Only files beneath a successful run are generated; there is no placeholder `outputs/matching_results.tsv` pretending to be a submission.

The internal validator checks full S1 coverage, ID membership, duplicate pairs/matches, candidate containment, and a TSV round-trip. **The official validator has not run because it has not been supplied.**

## Evaluation and baseline

Per-entity score is `1.25 * TP / (0.25 * number_of_true_matches + number_of_predictions)`. Empty truth plus empty predictions scores 1; false matches on a singleton score 0. Macro averaging includes every S1. Diagnostic precision and recall are micro link metrics; they do not select the model.

A seeded GroupShuffleSplit reserves 20% of S1 for held-out reporting. The remaining 80% calibrates the cosine threshold. No random pair split is used. TF-IDF fits unsupervised text within each train/test partition; no labels enter retrieval. A future supervised matcher needs an additional S1 training/calibration separation or out-of-fold training scores. Repeated selection on this holdout can overfit; establish fixed grouped folds before broader experiments.

Normalization uses Unicode NFKC, lowercase, punctuation-to-space, and whitespace collapse; raw fields remain intact. Accents and letters from all scripts are retained. No country filter or India/US dictionary is used.

Character TF-IDF (3–5 grams) retrieves up to 50 nonzero-overlap candidates independently from S2 and S3. Query batches avoid allocating the full dense S1×target matrix. A 300,000-feature cap and batch size are configurable. Large sparse products and Python candidate rows can still consume memory: benchmark a real-data sample before scaling. No-overlap and blank queries may have zero candidates. The matcher receives exactly the candidate pool later exported in the provisional candidate TSV.

Threshold calibration uses the requested coarse/fine sweep plus 0 and an above-1 predict-none option. Ties favor higher thresholds. Reported slices include singletons, matched and multi-match entities, observed countries, and S2/S3. Source slices include source-specific singletons. Pair recall and complete coverage are both measured; matched-only complete coverage avoids inflation from true singletons. Country transfer is currently a **threshold-transfer** check, not neural training: exact normalized `india` and `us` values are used for those two named checks, while all country strings remain usable in the pipeline.

## Portable GPU reranker workflow

Do this only after real-data EDA and the CPU baseline pass. Free GPU availability and duration are not assumed. No full neural operation is launched locally.

1. Export the exact baseline candidate pairs:

```sh
python scripts/03_export_gpu_pairs.py --config config.json --partition train --candidates artifacts/experiments/char_v1/train_candidates.tsv --output train_pairs.jsonl
```

2. Upload that file and `scripts/gpu/score_pairs.py` into a private Kaggle/Colab runtime if permitted by the competition's data rules. Enable a CUDA GPU. In that environment:

```sh
pip install torch transformers huggingface_hub sentencepiece
python score_pairs.py --pairs train_pairs.jsonl --output-dir reranker_pretrained --budget-seconds 3600
```

The script checks live Hugging Face model-card license metadata and safetensors parameter count before loading weights, pins the resolved repository revision, saves the repository README and verification evidence, checks loaded parameter count, then measures a sample spanning text-length quantiles. It projects full inference with a 50% margin and stops if over budget. It also enforces a full-inference time box; interrupted scores remain `.partial` and cannot be imported. Setup/download time is recorded separately. A CUDA OOM or other error must be investigated on the sample; do not repeatedly launch full jobs.

Only MIT/Apache-2.0 repository metadata and <=8B parameters are accepted automatically. Missing/ambiguous metadata stops the job for manual repository review. Read the saved source documentation and actual challenge licensing terms before accepting the experiment. Merge verified evidence into `artifacts/compliance/model_compliance.md`; repeat verification at final freeze. The local compliance file currently approves **no neural model**.

3. Download the complete GPU output directory and evaluate:

```sh
python scripts/04_evaluate_gpu_scores.py --config config.json --baseline-run artifacts/experiments/char_v1 --pairs train_pairs.jsonl --cache-dir reranker_pretrained --output artifacts/validation/reranker_pretrained.json
```

The importer verifies the export hash, exact candidate coverage, no duplicates, finite scores, and unchanged training files. It uses the baseline calibration/validation split. It reports a comparison; it does not promote a model automatically. GPU code is syntax-checked but has not been executed on a GPU. Test-score import/final neural packaging should be implemented only after a pretrained validation baseline succeeds.

## Next measured steps and time boxes

- Phase 0 (90 min): inspect all supplied TSVs and official rules/validator; confirm schema, label completeness, 0/1/many distribution, countries, missingness, and shared S2/S3 targets. The EDA script writes counts and examples. No global one-to-one rule is implemented.
- Baseline (2 h): sample benchmark, then run complete lexical retrieval, calibration and held-out evaluation; adapt official output serialization and run the official validator.
- Retrieval (3 h active debugging): add BM25 + RRF only if candidate recall/complete coverage expose a bottleneck; dense retrieval is deferred until a GPU benchmark supports it.
- Reranker (2 h integration): use the external script; keep only measured improvement with acceptable singleton/country/source behavior.
- Fine-tuning (3 h setup cap): **deferred** until the pretrained baseline exists. Use training-group hard negatives and singleton negatives; preserve v1 permanently. One mining round only initially.
- Optional features (90 min): add supervised calibration only with proper grouped training and measured gain.
- Final day: reserve at least 2 h for test inference, exact final candidate output, official validation, compliance re-verification, manifest, methodology and packaging.

Two serious failures in a component trigger a checkpoint and fallback. No unfinished neural or hybrid scaffolding replaces the working lexical route. Actual stage runtime estimates require real record/candidate counts and sample throughput; none are invented here.

## Submission control

`artifacts/submissions/submission_log.csv` is initialized and empty. There is no leaderboard API or automatic submission code. Explicit human approval is required for every upload. Before an approved upload, check the challenge calendar-day/timezone and count existing entries; refuse a sixth submission. Record date, ordinal, experiment, local score, description, returned leaderboard score and notes. Challenge timezone is still unknown.

## Methodology and ablations

See `artifacts/methodology.md`. The only implemented matching baseline is lexical cosine; no measured competition ablation is available. BM25/RRF, dense retrieval, supervised features, fine-tuning, mining and global consistency are deferred pending real evidence. A selected production run will be explicitly copied into the requested final manifest/output layout only after official validation; per-run manifests protect intermediate work now.
