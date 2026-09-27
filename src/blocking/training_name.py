"""Training-only business-name alias maps for a separate retrieval route.

Maps are built only from linked training pairs. An alias edge is admitted only
when it recurs across multiple distinct S1 and target entities.
"""
from collections import defaultdict
import hashlib
import json
from src.normalize import normalize

VERSION = "training-name-alias-components-v1"


def _name_key(value):
    return normalize(value)


def map_digest(alias_map):
    unsigned = {k: v for k, v in alias_map.items() if k != 'map_sha256'}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def verify_alias_map(alias_map):
    return alias_map.get('version') == VERSION and alias_map.get('map_sha256') == map_digest(alias_map)


def build_alias_map(s1_rows, s2_rows, s3_rows, truth, input_hashes, min_support=2):
    if min_support < 2:
        raise ValueError("min_support must be at least 2 distinct linked entities")
    s1 = {r['entity_id']: r for r in s1_rows}
    targets = {r['entity_id']: r for r in (*s2_rows, *s3_rows)}
    evidence = defaultdict(lambda: {'s1': set(), 'targets': set(), 'pairs': set()})
    for owner, matches in truth.items():
        if owner not in s1:
            raise ValueError(f"Unknown training S1 ID: {owner}")
        for target in matches:
            if target not in targets:
                raise ValueError(f"Unknown training target ID: {target}")
            left, right = _name_key(s1[owner]['business_name']), _name_key(targets[target]['business_name'])
            if not left or not right or left == right:
                continue
            edge = tuple(sorted((left, right)))
            evidence[edge]['s1'].add(owner)
            evidence[edge]['targets'].add(target)
            evidence[edge]['pairs'].add((owner, target))

    # Keep only recurring edges with independent entity support; no single linked
    # pair can create a retrieval mapping. Components allow multiple spelling
    # variants observed in separate training links to share one lookup key.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    approved = []
    for edge, proof in sorted(evidence.items()):
        if len(proof['s1']) < min_support or len(proof['targets']) < min_support:
            continue
        a, b = map(find, edge)
        if a != b:
            parent[max(a, b)] = min(a, b)
        approved.append((edge, proof))

    groups = defaultdict(list)
    for name in parent:
        groups[find(name)].append(name)
    aliases = {}
    for names in groups.values():
        canonical = min(names)
        for name in names:
            aliases[name] = canonical
    proofs = []
    used_s1, used_targets, used_pairs = set(), set(), set()
    for (left, right), proof in approved:
        used_s1.update(proof['s1']); used_targets.update(proof['targets']); used_pairs.update(proof['pairs'])
        proofs.append({'left': left, 'right': right, 'distinct_s1_count': len(proof['s1']),
                       'distinct_target_count': len(proof['targets']),
                       'training_s1_ids': sorted(proof['s1']),
                       'training_target_ids': sorted(proof['targets']),
                       'training_pairs': [list(p) for p in sorted(proof['pairs'])]})
    payload = {'version': VERSION, 'min_support': min_support,
               'input_sha256': dict(sorted(input_hashes.items())),
               'training_s1_ids': sorted(used_s1), 'training_target_ids': sorted(used_targets),
               'training_partition_s1_ids': sorted(truth),
               'training_pairs': [list(p) for p in sorted(used_pairs)],
               'aliases': dict(sorted(aliases.items())), 'evidence': proofs}
    payload['map_sha256'] = map_digest(payload)
    return payload


def iter_name_aliases(s1_rows, target_rows, alias_map):
    """Stream target-to-S1 candidates via exact alias component membership."""
    aliases = alias_map['aliases']
    support_by_name = defaultdict(lambda: 1)
    for proof in alias_map['evidence']:
        support = proof['distinct_s1_count'] + proof['distinct_target_count']
        support_by_name[proof['left']] = max(support_by_name[proof['left']], support)
        support_by_name[proof['right']] = max(support_by_name[proof['right']], support)
    source_index = defaultdict(list)
    for row in s1_rows:
        key = _name_key(row['business_name'])
        component = aliases.get(key, key)
        if component:
            source_index[component].append((row, support_by_name[key]))
    for target in target_rows:
        key = _name_key(target['business_name'])
        component = aliases.get(key, key)
        candidates = [(owner, min(support_by_name[key], owner_support))
                      for owner, owner_support in source_index.get(component, ())]
        candidates.sort(key=lambda item: (-item[1], item[0]['entity_id']))
        prior_score = None
        rank = 0
        for position, (owner, support) in enumerate(candidates, 1):
            if support != prior_score:
                rank = position
                prior_score = support
            yield {'source1_entity_id': owner['entity_id'],
                           'target_entity_id': target['entity_id'],
                           'target_source': target.get('target_source'),
                           'target_name': target['business_name'],
                           'target_address': target.get('business_address', ''),
                           'target_country': target.get('country', ''),
                           'retrieval_routes': ['training_name_alias'],
                           'name_alias_key': component, 'name_alias_support': support,
                           'name_alias_score': float(support), 'name_alias_rank': rank}


def retrieve_name_aliases(s1_rows, target_rows, alias_map):
    """Materializing convenience wrapper for fixtures and small runs."""
    return list(iter_name_aliases(s1_rows, target_rows, alias_map))
