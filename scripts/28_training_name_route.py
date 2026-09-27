"""Build, run, and evaluate a training-only business-name alias route."""
import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import chain
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blocking.training_name import VERSION, build_alias_map, iter_name_aliases, map_digest, verify_alias_map
from src.blocking.union import pair_key, unique_records
from src.evaluate import blocking_metrics


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tsv(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream, delimiter='\t')


def jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_truth(path, owners=None):
    truth = {}
    for row in tsv(path):
        if owners is not None and row['source1_entity_id'] not in owners:
            continue
        value = row['matched_entity_ids']
        truth[row['source1_entity_id']] = set(json.loads(value) if value.startswith('[')
                                              else value.split(',') if value else [])
    return truth


def build(args):
    start = time.monotonic()
    splits = json.loads(args.split.read_text())
    if 'train' not in splits:
        raise ValueError('split.json must contain a train partition')
    train_ids = set(splits['train'])
    other_ids = set().union(*(set(v) for k, v in splits.items() if k != 'train'))
    if train_ids & other_ids:
        raise ValueError('Training IDs overlap calibration/development/other partitions')
    truth = read_truth(args.labels, train_ids)
    if train_ids != truth.keys():
        raise ValueError('Training split contains IDs missing from labels')
    target_ids = set().union(*truth.values()) if truth else set()
    s1 = [r for r in tsv(args.s1) if r['entity_id'] in train_ids]
    s2 = [r for r in tsv(args.s2) if r['entity_id'] in target_ids]
    s3 = [r for r in tsv(args.s3) if r['entity_id'] in target_ids]
    if {r['entity_id'] for r in s1} != train_ids or ({r['entity_id'] for r in s2} | {r['entity_id'] for r in s3}) != target_ids:
        raise ValueError('Training labels reference missing S1 or target records')
    hashes = {name: sha(path) for name, path in [('s1', args.s1), ('s2', args.s2),
                                                  ('s3', args.s3), ('labels', args.labels),
                                                  ('split', args.split)]}
    mapping = build_alias_map(s1, s2, s3, truth, hashes, args.min_support)
    mapping['split_sha256'] = hashes['split']
    mapping['training_partition_s1_ids'] = sorted(train_ids)
    mapping['training_split_ids_sha256'] = hashlib.sha256('\n'.join(sorted(train_ids)).encode()).hexdigest()
    build_seconds = time.monotonic() - start
    mapping['map_sha256'] = map_digest(mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'version': VERSION, 'map_sha256': mapping['map_sha256'],
                      'approved_edges': len(mapping['evidence']), 'mapped_names': len(mapping['aliases']),
                      'training_s1_ids': len(mapping['training_s1_ids']),
                      'training_pairs': len(mapping['training_pairs']),
                      'input_sha256': hashes, 'seconds': build_seconds}))


def retrieve(args):
    start = time.monotonic()
    mapping = json.loads(args.map.read_text())
    if not verify_alias_map(mapping):
        raise ValueError('Unsupported or modified alias map')
    for name, path in (('s1', args.s1), ('s2', args.s2), ('s3', args.s3)):
        if mapping.get('input_sha256', {}).get(name) != sha(path):
            raise ValueError(f'Training map {name} input hash differs from retrieval input')
    wanted = {row['entity_id'] for row in jsonlines(args.sample)}
    if not wanted:
        raise ValueError('Sample is empty')
    sources = [r for r in tsv(args.s1) if r['entity_id'] in wanted]
    if {r['entity_id'] for r in sources} != wanted:
        raise ValueError('Sample contains unknown S1 IDs')
    targets = chain((dict(r, target_source=source) for r in tsv(path))
                    for source, path in [('s2', args.s2), ('s3', args.s3)])
    targets = chain.from_iterable(targets)
    args.output.mkdir(parents=True, exist_ok=False)
    count = 0
    with (args.output/'name_alias_pairs.jsonl').open('w', encoding='utf-8') as stream:
        for row in iter_name_aliases(sources, targets, mapping):
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    manifest = {'version': VERSION, 'complete': True, 'purpose': 'separate candidate route; no labels loaded',
                'map_sha256': mapping['map_sha256'],
        'inputs': {p.name: sha(p) for p in (args.s1, args.s2, args.s3, args.sample, args.map)},
        's1_count': len(sources), 'candidate_edges': count,
                'seconds': time.monotonic() - start}
    manifest['pairs_sha256'] = sha(args.output/'name_alias_pairs.jsonl')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


def evaluate_route(args):
    start = time.monotonic()
    split = json.loads(args.split.read_text())
    groups = {k: list(v) for k, v in split.items() if k.lower() in ('train', 'calibration', 'calib', 'development', 'dev')}
    if not groups:
        raise ValueError('No named train/calibration/development partitions in split file')
    samples = {r['entity_id']: r for r in jsonlines(args.sample)}
    truth_all = read_truth(args.labels)
    if set(samples) - truth_all.keys():
        raise ValueError('Sample IDs absent from labels')
    truth = {i: truth_all[i] for i in samples}
    base = unique_records(tsv(args.forward_pairs))
    forward = unique_records(r for r in jsonlines(args.forward_cache) if pair_key(r) in base)
    if base.keys() != forward.keys():
        raise ValueError('Forward raw cache does not cover frozen candidate TSV')
    if {owner for owner, _ in base} - samples.keys():
        raise ValueError('Forward pool contains an S1 outside the evaluation sample')
    route = list(unique_records(jsonlines(args.name_route/'name_alias_pairs.jsonl')).values())
    route_manifest = json.loads((args.name_route/'manifest.json').read_text())
    if not route_manifest.get('complete') or route_manifest.get('pairs_sha256') != sha(args.name_route/'name_alias_pairs.jsonl'):
        raise ValueError('Name route output is incomplete or its hash does not match')
    route_keys = {pair_key(r) for r in route}
    if {owner for owner, _ in route_keys} - samples.keys():
        raise ValueError('Name route contains an S1 outside the evaluation sample')
    pool = {}
    for key, row in forward.items():
        pool[key] = {**row, 'retrieval_routes': ['forward_structured'],
                     'forward_score': row.get('blocking_score'), 'forward_rank': row.get('blocking_rank')}
    for row in route:
        key = pair_key(row)
        if key in pool:
            pool[key]['retrieval_routes'].append('training_name_alias')
            for field in ('name_alias_key', 'name_alias_support', 'name_alias_score', 'name_alias_rank'):
                pool[key][field] = row[field]
        else:
            pool[key] = {**row, 'blocking_score': None, 'blocking_rank': None,
                         'forward_score': None, 'forward_rank': None}
    labels = truth
    base_sets = defaultdict(set); new_sets = defaultdict(set)
    for owner, target in base:
        base_sets[owner].add(target)
    for owner, target in pool:
        new_sets[owner].add(target)
    lost = set().union(*(base_sets[i] - new_sets[i] for i in samples))
    if lost:
        raise AssertionError('Union lost forward candidates')
    reports = {}
    for name, ids in groups.items():
        ids = [i for i in ids if i in samples]
        route_sets = defaultdict(set)
        for owner, target in route_keys:
            if owner in ids:
                route_sets[owner].add(target)
        before = blocking_metrics(labels, base_sets, ids)
        after = blocking_metrics(labels, new_sets, ids)
        route_only = blocking_metrics(labels, route_sets, ids)
        counts = [len(new_sets[i] - base_sets[i]) for i in ids]
        before_true = sum(len(labels[i] & base_sets[i]) for i in ids)
        after_true = sum(len(labels[i] & new_sets[i]) for i in ids)
        reports[name] = {'s1_count': len(ids), 'baseline': before,
                         'name_route_only': route_only, 'union': after,
                         'new_true_links': after_true-before_true, 'lost_true_links': 0,
                         'added_candidates_per_s1': dict(zip(('p50','p95','p99','max'),
                             np.quantile(counts, [.5,.95,.99,1]).tolist())) if counts else {},
                         'added_candidates': sum(counts)}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'forward_plus_name_alias.jsonl').open('w', encoding='utf-8') as stream:
        for key in sorted(pool):
            stream.write(json.dumps(pool[key], ensure_ascii=False) + '\n')
    report = {'purpose': 'exploratory candidate ablation only; no model-score claim',
              'route_version': VERSION,
              'inputs': {p.name: sha(p) for p in (args.sample,args.split,args.labels,args.forward_pairs,
                                                  args.forward_cache,args.name_route/'manifest.json')},
              'route_candidate_pairs': len(route_keys), 'forward_candidate_pairs': len(base),
              'union_candidate_pairs': len(pool), 'runtime_seconds_including_load_and_report': time.monotonic()-start,
              'route_generation_seconds': route_manifest['seconds'],
              'splits': reports}
    (args.output/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    b = commands.add_parser('build')
    for arg in ('s1','s2','s3','labels','split','output'):
        b.add_argument('--'+arg, type=Path, required=True)
    b.add_argument('--min-support', type=int, default=2)
    b.set_defaults(func=build)
    r = commands.add_parser('retrieve')
    for arg in ('s1','s2','s3','sample','map','output'):
        r.add_argument('--'+arg, type=Path, required=True)
    r.set_defaults(func=retrieve)
    e = commands.add_parser('evaluate')
    for arg in ('sample','split','labels','forward-pairs','forward-cache','name-route','output'):
        e.add_argument('--'+arg, type=Path, required=True)
    e.set_defaults(func=evaluate_route)
    parsed = parser.parse_args()
    parsed.func(parsed)
