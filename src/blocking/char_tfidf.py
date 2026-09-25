import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

COLUMNS = ["source1_entity_id", "target_entity_id", "target_source", "score", "rank"]

def retrieve(sources, k=50, batch_size=64, max_features=300000):
    """Independent S1->S2 and S1->S3 top K, with bounded sparse query batches.

    Unsupervised vocabulary sees the partition's records; labels never enter retrieval.
    Zero-overlap documents do not enter the candidate set. No country filtering.
    """
    if k < 1 or batch_size < 1:
        raise ValueError("k and batch_size must be positive")
    start = time.monotonic()
    text = [str(t) for f in sources.values() for t in f.retrieval_text]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32, max_features=max_features)
    if not any(len(t.strip()) >= 3 for t in text):
        return pd.DataFrame(columns=COLUMNS), {"runtime_seconds": time.monotonic()-start, "vocabulary_size": 0}
    vectorizer.fit(text)
    query = vectorizer.transform(sources["s1"].retrieval_text)
    rows = []
    qids = sources["s1"].entity_id.tolist()
    for source in ("s2", "s3"):
        targets = sources[source]
        if targets.empty:
            continue
        target_matrix = vectorizer.transform(targets.retrieval_text)
        tids = targets.entity_id.to_numpy()
        for offset in range(0, len(qids), batch_size):
            similarities = (query[offset:offset + batch_size] @ target_matrix.T).tocsr()
            for local in range(similarities.shape[0]):
                row = similarities.getrow(local)
                positive = row.data > 0
                indices, values = row.indices[positive], row.data[positive]
                order = np.lexsort((tids[indices], -values))[:k]
                for rank, at in enumerate(order, 1):
                    rows.append((qids[offset+local], tids[indices[at]], source, float(values[at]), rank))
    return pd.DataFrame(rows, columns=COLUMNS), {"runtime_seconds": time.monotonic()-start, "vocabulary_size": len(vectorizer.vocabulary_)}
