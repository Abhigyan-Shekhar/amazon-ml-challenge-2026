# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** BlackList  
**Team Members:** Parth Bhat, Abhigyan Shekhar, Mahima Koul  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We solve the business entity resolution task using a two-stage **structured blocking + supervised classification** pipeline. Candidate generation combines normalized name and address information with common-key pruning to reduce the comparison space, while an Apache-2.0 licensed XGBoost classifier uses lexical, structural, numeric, country, and fuzzy-similarity features to identify matching entities with a decision threshold optimized for macro F_0.5.

The final pipeline was designed around the challenge's precision-sensitive F_0.5 objective while preserving high candidate recall and explicitly handling missing address values.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis showed several recurring sources of difficulty:

- Business names can differ because of abbreviations, punctuation, token ordering, spelling variation, and partial names.
- Address strings contain formatting differences, shortened forms, house-number inconsistencies, and occasional missing values.
- Some true matches have limited exact-token overlap despite referring to the same entity.
- Multilingual and non-ASCII names introduce additional retrieval difficulty, particularly for Indic-script and Latin-diacritic variants.
- Very common blocking keys create large candidate sets while contributing limited discriminative value.
- Singleton entities are important because predicting an incorrect match directly hurts precision under the F_0.5 evaluation metric.

The selected larger-training experiment used 16,000 labeled S1 groups: 10,000 train, 2,000 calibration, 2,000 development, and 2,000 fresh confirmation. Its capped candidate recall was **92.238%** on development and **91.745%** on confirmation.

### 2.2 Solution Strategy

Our solution separates the problem into two stages:

1. Generate a relatively small candidate set for every Source-1 entity using structured name/address blocking.
2. Score each candidate pair using a supervised classifier trained on labeled entity pairs.

Candidate generation uses normalized structured information while limiting highly frequent keys. For every retrieved pair, we compute lexical, fuzzy, structural, numeric, country, and missing-value features. An XGBoost classifier then predicts a match probability, and the final probability threshold is selected using macro F_0.5 on the frozen calibration split.

**Approach Type:** Blocking + Classifier  
**Core Innovation:** Precision-oriented structured retrieval combined with capped common-key blocking, fuzzy comparison features, and explicit missing-address handling, with major retrieval and classification design decisions validated through controlled ablations.

---

## 3. Candidate Generation (Blocking)

Candidate generation reduces the otherwise impractical Cartesian comparison between Source-1 and Source-2/Source-3 records.

- **Blocking keys used:** Normalized business-name and address-derived structured keys, including token-based and numeric information. Highly frequent keys are pruned to prevent candidate explosion.
- **Maximum blocking-key frequency:** 10
- **Candidate cap:** Up to 20 candidates per target source for the capped retrieval configuration.
- **Full-test candidate pairs generated:** **44,931,896**
- **Measured candidate pair recall:** **92.238%** on development and **91.745%** on confirmation
- **Raw blocker recall in the 5,000-entity retrieval diagnostic:** approximately **92.65%**

**How we ensured true matches were not lost:**

Candidate-generation quality was measured independently of the classifier. We tracked pair-level candidate recall and analyzed missed labeled matches instead of relying only on final classifier performance.

The retrieval analysis identified multilingual spelling variation, transliteration, diacritics, and low lexical overlap as important causes of missed pairs. A Latin-diacritic folding ablation increased candidate-pair recall from **93.79% to 94.42%** on the 1,000-S1 development set and from **92.23% to 92.57%** on the disjoint 2,000-S1 confirmation set. However, the corresponding macro F_0.5 gain at the frozen 0.585 threshold was only **0.89215 to 0.89427** on development and **0.87741 to 0.87769** on confirmation. Because the end-to-end gain was small and not convincingly replicated, the folding variant was not promoted to the final pipeline.

---

## 4. Matching Model

**Features used:**

- **Name features:** Exact/normalized comparisons, token-level similarity, string-length information, and RapidFuzz-based fuzzy string similarities.
- **Address features:** Normalized address comparison, token similarity, fuzzy string similarity, numeric consistency, address-length information, and missing-address handling.
- **Other:** Country agreement, numeric comparison features, structured length features, and an explicit `address_missing` feature.

The selected classifier uses **16 engineered features**, including six RapidFuzz similarity features together with token, numeric, country, length, and missingness information.

**Model type:** XGBoost 3.2.0 `XGBClassifier` (Apache-2.0)

**Selected threshold:** **0.630**
**Threshold selection method:** Macro F_0.5 optimization on a grouped validation split, followed by evaluation on a fresh disjoint confirmation split.

Missing address comparisons are represented as missing numerical values while an explicit binary `address_missing` feature informs the classifier that the absence is structural rather than a low similarity score.

---

## 5. Results & Error Analysis

- **Development macro F_0.5:** **0.888485**
- **Development precision:** **0.966012**
- **Development recall:** **0.793169**
- **Fresh disjoint confirmation macro F_0.5:** **0.881313**
- **Fresh confirmation precision:** **0.963258**
- **Fresh confirmation recall:** **0.784493**

The full-test run produced:

- **Source-1 entities:** 1,732,544
- **Candidate pairs:** 44,931,896
- **Predicted matching links:** 5,236,137
- **Source-1 entities with no predicted match:** 128,414
- **Runtime:** 3,796.30 seconds (63.3 minutes)
- **Peak retrieval RSS:** 4.74 GiB

The generated outputs were checked using the official validator with full ID checking and produced **zero validation errors and zero warnings**.

**Common false positives (wrong merges):**

False positives were commonly associated with coincidental lexical overlap, multiple businesses sharing similar or identical address information, partial-name agreement, and cases where similar names referred to distinct business entities.

**Common false negatives (missed matches):**

False negatives were frequently associated with multilingual or transliterated names, Latin diacritics, substantial lexical variation, address-number inconsistencies, shortened addresses, and true matches with little exact-token overlap.

A dedicated analysis of raw blocking misses showed that Indic-script records represented a significant fraction of retrieval failures. Latin-diacritic folding improved retrieval coverage, but the confirmation-set F_0.5 improvement was only about **0.00028**, so the change was retained as an ablation result rather than promoted to production.

No labeled France training data were available, so no France F_0.5, precision, recall, or accuracy is reported. We instead performed an unlabeled domain-shift diagnostic comparing candidate and model-score distributions across France, India, and the United States.

---

## 6. Conclusion

The final selected system is the **10,000-group missing-address v2 structured-blocking + XGBoost pipeline at threshold 0.630**. XGBoost 3.2.0 is Apache-2.0 licensed and satisfies the challenge's stated model-license requirement. It preserves the validated retrieval and 16-feature missing-address design while replacing the historical scikit-learn classifier. The neural reranker and Latin-diacritic folding variant remain rejected experiments.

The project also demonstrated that entity-resolution performance depends strongly on retrieval quality: once a true entity pair is absent from the candidate set, the downstream classifier cannot recover it. Candidate recall and final macro F_0.5 were therefore treated as separate first-class evaluation metrics throughout development.

---

## Appendix

### A. Code Artefacts

The complete runnable solution is included under:

`code/business_entity_resolution/`

The packaged directory contains:

- `src/` — normalization, feature generation, candidate retrieval, model inference, and supporting pipeline code.
- `scripts/` — experiment, validation, profiling, and submission-generation scripts.
- `README.md` — reproduction and execution instructions.
- `requirements.txt` — Python dependencies required to reproduce the solution.
- `config.example.json` — example pipeline configuration.

The final packaging workflow generates:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Before packaging, the output files are checked with the official challenge validator, including full ID validation.

### B. Additional Results

#### Missing-address ablation

Changing missing address comparisons from ordinary similarity values to explicit missing numerical values, while retaining the `address_missing` indicator, produced:

- Validation macro F_0.5: **0.892146**
- Fresh confirmation macro F_0.5: **0.877411**

#### Latin-diacritic folding ablation

Latin-diacritic folding was tested only in retrieval keys and blocking ranks; the frozen v2 classifier inputs and threshold remained unchanged.

- Development candidate-pair recall: **0.93793 → 0.94422**
- Development macro F_0.5 at threshold 0.585: **0.89215 → 0.89427**
- Confirmation candidate-pair recall: **0.92234 → 0.92572**
- Confirmation macro F_0.5 at threshold 0.585: **0.87741 → 0.87769**

Calibration selected 0.615 for the folded candidate set, but that threshold was not evaluated on the disjoint confirmation set. The folded variant was therefore **not promoted**.

#### GPU reranker experiment

We also evaluated the Apache-2.0 licensed `BAAI/bge-reranker-v2-m3` pretrained reranker on the same validation setup.

Its macro F_0.5 was approximately **0.6482**, substantially below the selected CPU gradient-boosted classifier on the same evaluation split. It was therefore evaluated experimentally but was **not selected for the final matching pipeline**.

#### France domain-shift diagnostic

Because France has no labeled training examples, only unlabeled distribution diagnostics were performed.

For deterministic 1,000-entity samples:

- France average candidates per Source-1 entity: 28.491
- India: 27.384
- United States: 23.983

France therefore did not appear candidate-starved, although its candidate and model-score distributions differed from the labeled countries. No supervised France performance claim is made.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
