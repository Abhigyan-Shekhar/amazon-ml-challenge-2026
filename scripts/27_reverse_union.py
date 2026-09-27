"""Freeze forward/reverse top-1/3/8 unions and evaluate candidate ceilings.

Requires a completed FULL target scan, exact input hashes, and disjoint frozen
splits. Labels are consulted only after each label-independent union is formed.
Historical development results are exploratory, not an untouched confirmation.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blocking.union import pair_key, union_records, unique_records
from src.evaluate import blocking_metrics, evaluate


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def metrics(truth, candidates, ids, base):
    counts = [len(candidates.get(i, set())) for i in ids]
    oracle = {i: truth[i] & candidates.get(i, set()) for i in ids}
    recovered = sum(len(oracle[i] - base.get(i, set())) for i in ids)
    lost = sum(len((truth[i] & base.get(i, set())) - oracle[i]) for i in ids)
    return {**blocking_metrics(truth, candidates, ids),
            's1_count': len(ids), 'true_links': sum(len(truth[i]) for i in ids),
            'found_true_links': sum(len(oracle[i]) for i in ids),
            'new_true_links': recovered, 'lost_true_links': lost,
            'candidate_count': sum(counts),
            'additional_candidates': sum(counts) - sum(len(base.get(i, set())) for i in ids),
            'candidates_per_s1': dict(zip(('p50', 'p95', 'p99', 'max'),
                np.quantile(counts, [.5, .95, .99, 1]).tolist())) if counts else {},
            'oracle_macro_F0.5': evaluate(truth, oracle, ids)['macro_F0.5']}


def run(args):
    start = time.monotonic()
    manifest = json.loads((args.reverse_dir/'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('benchmark_only'):
        raise ValueError('Refusing partial/prefix benchmark as a full candidate evaluation')
    for path, expected in ((args.sample, manifest['sample_sha256']),
                           (args.forward_pairs, manifest['forward_pairs_sha256']),
                           (args.reverse_dir/'reverse_pairs.jsonl', manifest['pairs_sha256']),
                           (args.reverse_dir/'target_context.jsonl', manifest['context_sha256'])):
        if sha(path) != expected:
            raise ValueError('Input hash mismatch: ' + str(path))
    if manifest['k'] < max(args.ks):
        raise ValueError('Requested union k exceeds retrieved context k')
    samples = list(jsonlines(args.sample))
    queries = {r['entity_id']: r for r in samples}
    if len(queries) != len(samples):
        raise ValueError('Duplicate sample ID')
    splits = json.loads(args.split.read_text())
    members = [i for ids in splits.values() for i in ids]
    if len(members) != len(set(members)) or set(members) != set(queries):
        raise ValueError('Frozen splits must be disjoint and cover the sample')
    with args.forward_pairs.open(newline='') as f:
        frozen = unique_records(csv.DictReader(f, delimiter='\t'))
    if {k[0] for k in frozen} - queries.keys():
        raise ValueError('Unknown S1 in frozen forward pool')
    forward = unique_records(r for r in jsonlines(args.forward_cache) if pair_key(r) in frozen)
    if forward.keys() != frozen.keys():
        raise ValueError('Raw forward cache does not cover frozen forward pool')
    for key, row in forward.items():
        if row['target_source'] != frozen[key]['target_source']:
            raise ValueError('Frozen forward source mismatch')
    reverse = list(jsonlines(args.reverse_dir/'reverse_pairs.jsonl'))
    if any(r['source1_entity_id'] not in queries for r in reverse):
        raise ValueError('Unknown S1 in reverse pool')
    if any(not 1 <= r['reverse_rank'] <= manifest['k'] or
           not r['reverse_score'] > manifest['score_floor_strict'] for r in reverse):
        raise ValueError('Reverse record violates manifest rank/floor')
    context_rows = list(jsonlines(args.reverse_dir/'target_context.jsonl'))
    contexts = {r['target_entity_id']: r for r in context_rows}
    if len(contexts) != len(context_rows):
        raise ValueError('Duplicate target context')
    base = defaultdict(set)
    for owner, target in forward:
        base[owner].add(target)
    truth = {i: set(r['matches']) for i, r in queries.items()}
    groups = {'all': list(queries), **splits}
    for name, ids in list(groups.items()):
        for country in sorted({queries[i]['country'] for i in ids}):
            groups[name+'/country:'+country] = [i for i in ids if queries[i]['country'] == country]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'purpose': 'Historical exploratory retrieval evaluation; not model/leaderboard accuracy',
              'reverse_manifest': manifest, 'inputs': {name: sha(path) for name, path in
                [('sample', args.sample), ('split', args.split), ('forward_pairs', args.forward_pairs),
                 ('forward_cache', args.forward_cache), ('reverse_manifest', args.reverse_dir/'manifest.json')]},
              'baseline': {name: metrics(truth, base, ids, base) for name, ids in groups.items()},
              'unions': {}}
    for k in sorted(set(args.ks)):
        candidates = defaultdict(set)
        routes = defaultdict(int)
        path = args.output/f'union_k{k}.jsonl'
        with path.with_suffix('.jsonl.partial').open('x') as stream:
            for row in union_records(forward.values(), reverse, contexts, k):
                candidates[row['source1_entity_id']].add(row['target_entity_id'])
                routes['+'.join(row['retrieval_routes'])] += 1
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        path.with_suffix('.jsonl.partial').rename(path)
        if any(not values <= candidates[owner] for owner, values in base.items()):
            raise AssertionError('Union dropped a forward candidate')
        report['unions'][str(k)] = {'path': path.name, 'sha256': sha(path), 'routes': dict(routes),
                                   'metrics': {name: metrics(truth, candidates, ids, base) for name, ids in groups.items()}}
    report.update(complete=True, seconds=time.monotonic()-start)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sample', 'split', 'forward-pairs', 'forward-cache', 'reverse-dir', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--ks', type=int, nargs='+', default=[1, 3, 8])
    a = p.parse_args()
    if any(k < 1 for k in a.ks):
        p.error('ks must be positive')
    run(a)
