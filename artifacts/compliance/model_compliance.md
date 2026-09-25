# Model compliance

Current CPU models use no pretrained neural weights and no external factual business data. Logistic regression has 11 fitted parameters for the token-only fallback; preprocessing and classifiers use scikit-learn. No neural model has been loaded or trained.

## Pre-experiment repository verification — 2026-09-25 UTC

### BAAI/bge-reranker-v2-m3

- Source: https://huggingface.co/api/models/BAAI/bge-reranker-v2-m3
- Revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- Repository license: **apache-2.0**
- Parameter count: 567755777
- Count method: Hugging Face repository safetensors metadata
- Verified at: 2026-09-25T14:59:26.197862+00:00

### BAAI/bge-m3

- Source: https://huggingface.co/api/models/BAAI/bge-m3
- Revision: `5617a9f61b028005a4858fdac845db406aefb181`
- Repository license: **mit**
- Parameter count: not available in metadata; NOT approved for use yet
- Count method: Hugging Face repository safetensors metadata
- Verified at: 2026-09-25T14:59:26.473467+00:00

BAAI/bge-reranker-v2-m3 satisfies the stated Apache-2.0 and <=8B gates based on repository metadata (567,755,777 parameters). The GPU script repeats this check before model loading and records loaded parameter count. Review the challenge-specific model rules when supplied; only the dataset and output validator have been supplied so far.

BGE-M3 remains deferred: MIT metadata was verified, but its parameter-count evidence is incomplete and it is not used.

Repeat repository/license/count verification before neural final packaging. `repository_evidence.json` is the saved dated response summary. No neural final packaging or inference has occurred.

## Selected CPU checkpoint

The selected `capped10_fuzzy_tree` is a scikit-learn HistGradientBoosting classifier with 10 token features and 6 RapidFuzz edit features. It uses no pretrained weights and no external data. Installed distribution metadata and tree-size evidence are saved in `cpu_final_evidence.json`. Neural <=8B licensing restrictions are not being used to approve an unverified neural model. The verified reranker remains unexecuted.
