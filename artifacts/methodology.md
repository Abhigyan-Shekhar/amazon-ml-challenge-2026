# Measured methodology — 2026-09-25

All seven provided TSVs were audited without retaining record text in memory. They contain 2,206,821 training references and 10,320,219 training targets; test contains 1,732,544 references and 9,969,589 targets. Ground truth covers every training S1 and no S2/S3 truth ID maps to multiple S1s. No external factual data or neural inference is used.

Global character TF-IDF retrieval failed the local feasibility gate: a sampled sparse-matrix estimate alone consumes nearly all 16 GiB of RAM, with a rough several-day full-query estimate. A single-name-token fallback also produced too many comparisons and was stopped. Label-independent structured name blocks were then measured against all training targets for a uniform 5,000-S1 reservoir. Address blocks were added only after observed link recall was inadequate. This increased full sample link recall from 75.29% to 93.00% at 100 candidates/source.

Normalized text uses Unicode NFKC, lowercase and punctuation spaces. Raw records remain unchanged. There is no hard country filter. Within-block character TF-IDF and token/numeric/country/length features feed logistic regression; a bounded shallow HistGradientBoosting experiment uses the same 14 features and no random pair holdout. The split is 3,000 training, 1,000 threshold-calibration, 1,000 held-out S1s. The effective candidate cap is 50/source for the initial name-only matcher and 100/source for the address and exact fallback experiments. The name-to-address comparison also changes K; it is not a pure address-only ablation. Thresholds and model retention use calibration groups.

| Change | Held-out macro F0.5 |
|---|---:|
| Structured name retrieval + cosine | 0.690951 |
| + logistic lexical features | 0.771937 |
| + address retrieval | 0.821921 |
| + shallow boosted-tree matcher | **0.874415** |
| Independent exact-core/address CPU fallback + cheap logistic | 0.690681 |

The earlier character-feature tree has 96.13% link precision, 77.52% link recall, 90.38% singleton F0.5 and 93.79% candidate link recall on held-out S1s. India and US F0.5 are 0.833815 and 0.901936. Bootstrap sampling interval is approximately [0.8591, 0.8878]; it excludes selection and domain-shift uncertainty. There is no France validation. Country-transfer logistic scores are 0.838154 India→US and 0.726695 US→India.

The selected deployed model uses ten token/numeric/country/length features and six RapidFuzz similarities, with a depth-4 HistGradientBoosting classifier (200 iterations) and frozen threshold 0.64. Retrieval excludes blocking keys shared by more than ten S1s and retains up to 20 candidates per source. Development macro F0.5 is 0.890226. A fresh disjoint 2,000-S1 confirmation set, without threshold retuning, scores 0.875554, with 0.9588 link precision, 0.7839 recall and 0.9223 candidate recall. These are local validation scores, not leaderboard scores.

Full-test fuzzy inference completed for all 1,732,544 S1s, scoring 44,931,896 candidates and selecting 5,196,796 links in 2,165.16 seconds. Peak measured retrieval RSS was 7.55 GiB. The package's official_validation.json records the subsequent official ID and format validation. No leaderboard upload has been performed.

The full-test fallback is a separate runtime-feasible checkpoint. Its output must not be presented as predictions from the stronger sample model. It retrieves by exact sorted name-core or exact sorted address tokens, then token-overlap top K per source, then frozen token-feature logistic inference. Both output files include every S1, including blank singleton fields. Candidate output is generated from the finalized pool actually scored.

The GPU package exports the fixed development candidate pool (227,362 pairs), with split metadata. Repository evidence was checked for the intended reranker; runtime, inference performance and fine-tuning are still unmeasured. Dense retrieval, BM25/RRF, neural fine-tuning and global consistency are not claimed as completed ablations. Existing successful checkpoints are preserved. The next priorities are improving retrieval misses and running the pretrained reranker benchmark.
