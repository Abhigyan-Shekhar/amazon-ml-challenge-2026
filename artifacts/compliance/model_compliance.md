# Model compliance

Current CPU models use no pretrained neural weights and no external factual business data. Logistic regression has 11 fitted parameters for the token-only fallback; preprocessing and classifiers use scikit-learn. No neural model is used in the selected CPU matching pipeline. BAAI/bge-reranker-v2-m3 was loaded and executed only for a controlled validation benchmark; no neural model was trained or fine-tuned.

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

BAAI/bge-reranker-v2-m3 satisfies the stated Apache-2.0 and <=8B gates based on repository metadata (567,755,777 parameters). The GPU script repeats this check before model loading and records loaded parameter count. The challenge rules require attention to model license and parameter-count constraints. The BAAI/bge-reranker-v2-m3 satisfies the recorded Apache-2.0 and <=8B checks, but it was not selected because its measured validation performance was substantially below the CPU matcher.

BGE-M3 remains deferred: MIT metadata was verified, but its parameter-count evidence is incomplete and it is not used.

Repeat repository/license/count verification before neural final packaging. `repository_evidence.json` is the saved dated response summary. No neural model has been selected or packaged for the final pipeline. Experimental pretrained reranker inference was performed only for validation and was rejected.

## Selected CPU checkpoint

The selected `capped10_fuzzy_tree` is a scikit-learn HistGradientBoosting classifier with 10 token features and 6 RapidFuzz edit features. It uses no pretrained weights and no external data. Installed distribution metadata and tree-size evidence are saved in `cpu_final_evidence.json`. Neural <=8B licensing restrictions are not being used to approve an unverified neural model. The verified reranker was executed on the fixed validation pool and rejected after scoring approximately 0.6482 macro F0.5 versus approximately 0.8902 for the selected fuzzy CPU tree.

### Final CPU model licensing note

The selected `capped10_fuzzy_tree` is implemented with scikit-learn and RapidFuzz and uses no pretrained model weights or external business data. scikit-learn is distributed under the BSD-3-Clause license. Because the challenge wording states that the final model should be MIT/Apache-2.0 licensed and <=8B parameters, the team should obtain organizer clarification on whether that licensing clause is intended to apply to locally trained classical estimators and their software libraries before final submission.
