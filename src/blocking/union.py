"""Lossless union of frozen forward candidates and ranked reverse candidates.

This module never reads labels. Missing reverse pair scores/ranks remain unknown,
not zero; a complete top-k target context is not a complete competition graph.
"""


def pair_key(row):
    return row['source1_entity_id'], row['target_entity_id']


def unique_records(records):
    result = {}
    for row in records:
        key = pair_key(row)
        if key in result:
            raise ValueError('Duplicate candidate pair: ' + str(key))
        result[key] = row
    return result


def union_records(forward, reverse, contexts, k):
    if k < 1:
        raise ValueError('k must be positive')
    pool = {}
    for key, row in unique_records(forward).items():
        pool[key] = {**row, 'retrieval_routes': ['forward_structured'],
                     'forward_score': row.get('blocking_score'),
                     'forward_rank': row.get('blocking_rank')}
    for key, row in unique_records(reverse).items():
        if row['reverse_rank'] > k:
            continue
        if key in pool:
            for field in ('target_source', 'target_name', 'target_address', 'target_country'):
                if pool[key][field] != row[field]:
                    raise ValueError('Forward/reverse target mismatch: ' + str(key))
            pool[key].update({field: value for field, value in row.items() if field.startswith('reverse_')})
            pool[key]['retrieval_routes'].append('reverse_sparse')
        else:
            pool[key] = {**row, 'retrieval_routes': ['reverse_sparse'],
                         'blocking_score': None, 'blocking_rank': None,
                         'forward_score': None, 'forward_rank': None}
    for key in sorted(pool):
        row = pool[key]
        if key[1] not in contexts:
            raise ValueError('Missing full-S1 target context: ' + key[1])
        context = contexts[key[1]]
        row['reverse_best_s1_id'] = context['best_s1_id']
        row['reverse_best_owner_score'] = context['best_score']
        row['reverse_second_owner_score'] = context['second_score']
        own = next((o for o in context['top_owners'] if o['source1_entity_id'] == key[0]), None)
        row['reverse_score'] = own['score'] if own else None
        row['reverse_rank'] = own['rank'] if own else None
        # Absent from context does not mean zero similarity or rank k+1.
        row['reverse_rank_censored'] = own is None
        row['reverse_gap_to_best'] = context['best_score'] - own['score'] if own else None
        other = context['second_score'] if context['best_s1_id'] == key[0] else context['best_score']
        row['reverse_margin_to_best_other'] = own['score'] - other if own and other is not None else None
        yield row
