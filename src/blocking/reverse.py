"""Label-independent sparse reverse retrieval with a full-S1 competitor index.

The route ranks each target against ALL reference S1s. An optional emission
subset only gates targets with no positive score above the route's score floor
against that subset; it never changes the competitor population or IDF.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.normalize import normalize

VERSION = 'reverse-char3-address-word-v1'


def vectorizers():
    return (TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2,
                           max_df=.05, sublinear_tf=True, dtype=np.float32, lowercase=False),
            TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', min_df=2,
                           max_df=.05, sublinear_tf=True, dtype=np.float32, lowercase=False))


def combine(name, address, name_weight=.5):
    if not 0 < name_weight < 1:
        raise ValueError('name_weight must be strictly between 0 and 1')
    # Missing fields contribute zero; do not renormalize the concatenation.
    return sparse.hstack([name.multiply(np.float32(np.sqrt(name_weight))),
                          address.multiply(np.float32(np.sqrt(1-name_weight)))], format='csr', dtype=np.float32)


def transform(rows, name_vec, address_vec, name_weight=.5):
    return combine(name_vec.transform([normalize(r['business_name']) for r in rows]),
                   address_vec.transform([normalize(r['business_address']) for r in rows]), name_weight)


def topk(matrix, index_transposed, k, floor=0., threads=2):
    if k < 1 or threads < 1 or not 0 <= floor < 1:
        raise ValueError('Require k/threads >=1 and 0 <= floor <1')
    return sp_matmul_topn(matrix, index_transposed, top_n=min(k, index_transposed.shape[1]),
                         threshold=floor, sort=True, n_threads=threads).tocsr()


@dataclass
class ReverseBatch:
    hits: list
    targets_scored_globally: int
    contexts: list


def retrieve_batch(targets, full_index_transposed, k=8, floor=.35, threads=2,
                   emit_indices=None, emit_index_transposed=None, force_rows=(),
                   witness_index_transposed=None):
    """Return top-k positive-score owners and top-two competitor context.

    Score comparisons are strictly > floor. Ties have competition rank (one plus
    the number of strictly better scores), making top1/3/8 nested and tie-aware.
    All ties at the top-k boundary are retained, not arbitrarily cut by backend
    order. Thus k is a rank budget, not an absolute number of output owners.
    Context describes all-S1 ranking under this scorer, not model probabilities.
    """
    if emit_indices is not None and emit_index_transposed is None:
        raise ValueError('Emission subset requires its gate matrix')
    selected = np.arange(targets.shape[0])
    if emit_indices is not None:
        gate = topk(targets, emit_index_transposed, 1, floor, threads)
        selected = np.union1d(np.flatnonzero(np.diff(gate.indptr)), np.asarray(force_rows, dtype=int))
    if witness_index_transposed is not None and emit_indices is not None and len(selected):
        # A strict lower bound on the number of better global competitors.
        # Equality never prunes: boundary ties must survive. Forced contexts bypass it.
        witness = topk(targets[selected], witness_index_transposed, k, 0., threads)
        sample_best = np.asarray(gate.max(axis=1).toarray()).ravel()
        forced = set(force_rows)
        keep = []
        for local, target_row in enumerate(selected):
            values = witness.data[witness.indptr[local]:witness.indptr[local+1]]
            if target_row in forced or len(values) < k or values.min() <= sample_best[target_row]:
                keep.append(target_row)
        selected = np.asarray(keep, dtype=int)
    if not len(selected):
        return ReverseBatch([], 0, [])
    subset = targets[selected]
    # Extra result detects boundary ties; competitors below floor still inform gaps.
    scores = topk(subset, full_index_transposed, max(k+1, 2), 0., threads)
    emit = None if emit_indices is None else set(emit_indices)
    hits, contexts = [], []
    for local, target_row in enumerate(selected):
        row = scores.getrow(local)
        ids, values = row.indices, row.data
        if not len(values):
            contexts.append({'target_row':int(target_row), 'best_owner_row':None,
                             'best_score':None,'second_score':None,'top_owners':[]})
            continue
        order = np.lexsort((ids, -values))
        ids, values = ids[order], values[order]
        if len(values) > k and values[k-1] == values[k]:
            # Sparse single-row product only for boundary ties: includes every tied owner.
            exact = (subset.getrow(local) @ full_index_transposed).tocsr()
            positive = exact.data > 0
            ids, values = exact.indices[positive], exact.data[positive]
            order = np.lexsort((ids, -values))
            ids, values = ids[order], values[order]
        best = float(values[0])
        second = float(values[1]) if len(values) > 1 else None
        context_owners=[]
        for pos, (owner, value) in enumerate(zip(ids, values)):
            rank = pos + 1 if pos == 0 or value < values[pos-1] else rank
            if rank>k:break
            context_owners.append({'owner_row':int(owner),'score':float(value),'rank':rank})
        contexts.append({'target_row':int(target_row),'best_owner_row':int(ids[0]),
                         'best_score':best,'second_score':second,'top_owners':context_owners})
        for pos, (owner, value) in enumerate(zip(ids, values)):
            rank = pos + 1 if pos == 0 or value < values[pos-1] else rank
            if rank > k or value <= floor:
                break
            if emit is not None and int(owner) not in emit:
                continue
            competitor = second if pos == 0 else best
            hits.append({'target_row': int(target_row), 'owner_row': int(owner),
                         'reverse_score': float(value), 'reverse_rank': rank,
                         'reverse_best_owner_score': best, 'reverse_second_owner_score': second,
                         'reverse_gap_to_best': best-float(value),
                         'reverse_margin_to_best_other': None if competitor is None else float(value)-competitor})
    return ReverseBatch(hits, len(selected), contexts)
