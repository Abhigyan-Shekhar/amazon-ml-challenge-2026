"""Label-free, versioned target competition features for finalized unions."""
import numpy as np

VERSION = 'target-competition-v1'
NUMERIC_NAMES = (
    'forward_score', 'forward_rank', 'reverse_score', 'reverse_rank',
    'reverse_best_owner_score', 'reverse_second_owner_score',
    'reverse_gap_to_best', 'reverse_margin_to_best_other',
)
FEATURE_NAMES = list(NUMERIC_NAMES) + [
    'reverse_rank_censored', 'route_forward', 'route_reverse', 'reverse_best_is_self',
]


def transform(rows):
    """Keep null numeric evidence as NaN, including censored pair scores/ranks.

    Routes are explicit; a non-null reverse rank does not imply reverse routing.
    IDs are used only to compare owner identity, never as model input categories.
    """
    out = np.full((len(rows), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(NUMERIC_NAMES):
            value = row.get(name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError('Nonfinite competition metadata: ' + name)
                if name.endswith('_rank') and (value < 1 or value != int(value)):
                    raise ValueError('Ranks must be positive integers or null')
                out[i, j] = value
        censored = row.get('reverse_rank_censored')
        if censored is not None:
            if not isinstance(censored, bool):
                raise ValueError('Censor flag must be boolean')
            if censored and any(row.get(k) is not None for k in
                                ('reverse_score', 'reverse_rank', 'reverse_gap_to_best',
                                 'reverse_margin_to_best_other')):
                raise ValueError('Censored reverse evidence must remain unknown')
            out[i, 8] = censored
        routes = row.get('retrieval_routes')
        if routes is not None:
            if not routes or set(routes) - {'forward_structured', 'reverse_sparse'}:
                raise ValueError('Unknown/empty retrieval routes')
            out[i, 9:11] = ['forward_structured' in routes, 'reverse_sparse' in routes]
        best = row.get('reverse_best_s1_id')
        if best is not None:
            out[i, 11] = best == row['source1_entity_id']
    return out
