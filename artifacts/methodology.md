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

The strongest sample model has 96.13% link precision, 77.52% link recall, 90.38% singleton F0.5 and 93.79% candidate link recall on held-out S1s. India and US F0.5 are 0.833815 and 0.901936. Bootstrap sampling interval is approximately [0.8591, 0.8878]; it excludes selection and domain-shift uncertainty. There is no France validation. Country-transfer logistic scores are 0.838154 India→US and 0.726695 US→India.

The full-test fallback is a separate runtime-feasible checkpoint. Its output must not be presented as predictions from the stronger sample model. It retrieves by exact sorted name-core or exact sorted address tokens, then token-overlap top K per source, then frozen token-feature logistic inference. Both output files include every S1, including blank singleton fields. Candidate output is generated from the finalized pool actually scored.

The GPU package exports only the exact held-out candidate pool (227,362 pairs). Repository evidence was checked for the intended reranker; runtime, inference performance and fine-tuning are still unmeasured. Dense retrieval, BM25/RRF, neural fine-tuning and global consistency are not claimed as completed ablations. Existing successful checkpoints are preserved. The next priority is scalable deployment of the stronger pipeline and improvement of retrieval misses, followed by the pretrained reranker benchmark.
