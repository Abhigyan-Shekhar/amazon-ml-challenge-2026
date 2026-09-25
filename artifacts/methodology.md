# Methodology — preliminary implementation

Conservative Unicode normalization preserves raw names and addresses. Independent character n-gram TF-IDF searches against S2 and S3 produce the final candidate union. Cosine similarity and an S1-group-calibrated threshold allow zero, one or many matches. An untouched S1 holdout measures exact macro F0.5 and diagnostic slices. No external factual data is incorporated.

| Change | Real macro F0.5 | Pair recall | Complete coverage | Status |
|---|---:|---:|---:|---|
| Character TF-IDF baseline | — | — | — | Implemented; synthetic correctness tests only |
| Sparse/BM25 + RRF | — | — | — | Deferred until recall measured |
| Dense retrieval | — | — | — | Deferred until compute/recall gate |
| Pretrained reranker | — | — | — | Portable scoring job prepared; unexecuted |
| Fine-tuned reranker | — | — | — | Requires successful pretrained baseline |
| Model-mined hard negatives | — | — | — | Requires validated v1 |
| Optional pair features | — | — | — | Requires measured justification |

No conclusions about competition performance are possible until the actual dataset and official evaluation assets arrive. No France validation is claimed. No one-to-one/global assignment assumption is made.
