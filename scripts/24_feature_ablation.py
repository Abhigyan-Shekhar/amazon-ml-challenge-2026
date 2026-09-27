"""Controlled feature-family ablations on a frozen candidate pool and S1 split.

Development is exploratory and may select configurations; it is not a lockbox.
No confirmation/test data is read. Output directories must be new.
"""
import argparse
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import evaluate, entity_f05, blocking_metrics
from src.features.matching import PairFeatureExtractor, BASE_NAMES, FAMILY_NAMES, VERSION, nonlatin
from src.predict import predict, select_threshold


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_inputs(sample_path, pairs_path, split_path, allowed_path=None):
    samples = [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
    queries = {r['entity_id']: r for r in samples}
    if len(queries) != len(samples):
        raise ValueError('Duplicate sample S1')
    split = json.loads(split_path.read_text())
    required = ('train', 'calibration', 'validation')
    if any(not split.get(k) for k in required):
        raise ValueError('Nonempty train/calibration/validation splits required')
    all_ids = [i for k in required for i in split[k]]
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(queries):
        raise ValueError('Splits must be disjoint and exactly cover sample S1s')
    allowed = None
    if allowed_path:
        frame = pd.read_csv(allowed_path, sep='\t', dtype=str)
        keys = list(zip(frame.source1_entity_id, frame.target_entity_id))
        allowed = set(keys)
        if len(keys) != len(allowed):
            raise ValueError('Duplicate frozen candidate pair')
    pairs, seen = [], set()
    with pairs_path.open() as stream:
        for line in stream:
            r = json.loads(line)
            key = (r['source1_entity_id'], r['target_entity_id'])
            if allowed is not None and key not in allowed:
                continue
            if key[0] not in queries or key in seen:
                raise ValueError('Unknown S1 or duplicate candidate pair')
            seen.add(key)
            pairs.append(r)
    if not pairs or (allowed is not None and seen != allowed):
        raise ValueError('Candidate cache does not cover the frozen pool')
    return queries, pairs, split


def paired_interval(truth, base, proposed, ids, seed=2026):
    delta = np.array([entity_f05(truth[i], proposed.get(i, set())) -
                      entity_f05(truth[i], base.get(i, set())) for i in ids])
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(2000)])
    return {'delta': float(delta.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'replicates': 2000, 'unit': 'S1', 'seed': seed,
            'caveat': 'Exploratory multiple comparisons; not independent confirmation.'}


def slices(truth, predicted, ids, queries, missing_target_ids):
    groups = {f'country:{c}': [i for i in ids if queries[i]['country'] == c]
              for c in sorted({queries[i]['country'] for i in ids})}
    groups.update({f'true_matches:{n}': [i for i in ids if
                   (len(truth[i]) == int(n) if n != '2+' else len(truth[i]) >= 2)]
                   for n in ('0', '1', '2+')})
    groups['nonlatin_s1_name'] = [i for i in ids if nonlatin(queries[i]['business_name'])]
    groups['missing_s1_address'] = [i for i in ids if not queries[i]['business_address'].strip()]
    groups['any_candidate_missing_address'] = [i for i in ids if i in missing_target_ids]
    return {k: {'s1': len(v), **evaluate(truth, predicted, v)} for k, v in groups.items()}


def run(args):
    queries, pairs, split = load_inputs(args.sample, args.pairs, args.split, args.allowed)
    args.output.mkdir(parents=True, exist_ok=False)
    train_rows = [queries[i] for i in split['train']]
    extractor = PairFeatureExtractor(tuple(FAMILY_NAMES)).fit(train_rows)
    start = time.perf_counter()
    x = np.vstack([extractor.transform(pairs[i:i+4096], queries) for i in range(0, len(pairs), 4096)])
    feature_seconds = time.perf_counter() - start
    frame = pd.DataFrame([{k: r[k] for k in ('source1_entity_id', 'target_entity_id', 'target_source')} for r in pairs])
    truth = {i: set(r['matches']) for i, r in queries.items()}
    y = np.array([r['target_entity_id'] in truth[r['source1_entity_id']] for r in pairs], dtype=np.int8)
    train = frame.source1_entity_id.isin(split['train']).to_numpy()
    params = dict(n_estimators=200, max_depth=4, learning_rate=.08, reg_lambda=1,
                  tree_method='hist', device='cpu', n_jobs=2, random_state=2026,
                  objective='binary:logistic', eval_metric='logloss')
    configurations = [c for n in range(4) for c in combinations(FAMILY_NAMES, n)]
    report = {'status': 'EXPLORATORY_NOT_PRODUCTION', 'feature_version': VERSION,
              'split_sizes': {k: len(v) for k, v in split.items()},
              'candidate_pairs': len(pairs), 'train_pairs': int(train.sum()),
              'model_params': params, 'full_feature_build_seconds': feature_seconds,
              'versions': {k: version(k) for k in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'rapidfuzz', 'anyascii')},
              'python': platform.python_version(), 'platform': platform.platform(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'implementation_sha256': {p: sha(Path(__file__).resolve().parents[1] / p) for p in
                                        ('src/features/matching.py', 'scripts/24_feature_ablation.py')},
              'inputs': {k: {'name': p.name, 'sha256': sha(p)} for k, p in
                         [('sample', args.sample), ('pairs', args.pairs), ('split', args.split)] +
                         ([('allowed', args.allowed)] if args.allowed else [])},
              'df_policy': 'Unique training S1 records only; no labels, held-out rows or target rows.',
              'runs': {}}
    pool = frame.groupby('source1_entity_id').target_entity_id.agg(set).to_dict()
    report['development_candidate_metrics'] = blocking_metrics(truth, pool, split['validation'])
    report['development_oracle_macro_f05'] = evaluate(truth, {i: truth[i] & pool.get(i, set()) for i in truth}, split['validation'])['macro_F0.5']
    missing_target_ids = {p['source1_entity_id'] for p in pairs if not p['target_address'].strip()}
    baseline = None
    for families in configurations:
        label = '+'.join(families) or 'baseline'
        run_dir = args.output / label
        run_dir.mkdir()
        selected = PairFeatureExtractor(families).fit(train_rows)
        columns = [extractor.feature_names.index(n) for n in selected.feature_names]
        values = x[:, columns]
        model = XGBClassifier(**params)
        start = time.perf_counter()
        model.fit(values[train], y[train])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        frame['score'] = model.predict_proba(values)[:, 1]
        prediction_seconds = time.perf_counter() - start
        threshold, sweep = select_threshold(frame, truth, split['calibration'])
        predicted = predict(frame, threshold)
        if baseline is None:
            baseline = predicted
        selected.save(run_dir / 'extractor.json')
        model.save_model(run_dir / 'model.json')
        # Check shared inference with a serialized extractor/model on a real batch.
        restored = PairFeatureExtractor.load(run_dir / 'extractor.json')
        probe = restored.transform(pairs[:100], queries)
        np.testing.assert_allclose(probe, values[:100], equal_nan=True)
        loaded = XGBClassifier()
        loaded.load_model(run_dir / 'model.json')
        np.testing.assert_allclose(loaded.predict_proba(probe)[:, 1], frame.score.to_numpy()[:100])
        benchmark_rows = pairs[:min(10000, len(pairs))]
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            selected.transform(benchmark_rows, queries)
            timings.append(time.perf_counter() - start)
        result = {'families': list(families), 'feature_count': len(columns),
                  'feature_names': selected.feature_names, 'threshold': float(threshold),
                  'calibration': evaluate(truth, predicted, split['calibration']),
                  'development': evaluate(truth, predicted, split['validation']),
                  'development_slices': slices(truth, predicted, split['validation'], queries, missing_target_ids),
                  'paired_vs_baseline': paired_interval(truth, baseline, predicted, split['validation']),
                  'fit_seconds': fit_seconds, 'prediction_seconds': prediction_seconds,
                  'feature_benchmark': {'pairs': len(benchmark_rows), 'seconds': timings,
                                        'median_pairs_per_second': len(benchmark_rows) / float(np.median(timings)),
                                        'scope': 'Feature transform only, warm cache, 3 repetitions; excludes I/O/retrieval.'}}
        metadata = {'feature_version': VERSION, 'feature_names': selected.feature_names,
                    'threshold': float(threshold), 'model_sha256': sha(run_dir / 'model.json'),
                    'extractor_sha256': sha(run_dir / 'extractor.json'),
                    'train_ids_sha256': selected.train_ids_sha256, 'model_params': params,
                    'versions': report['versions'], 'inputs': report['inputs'],
                    'production_promoted': False,
                    'candidate_contract': 'Score finalized candidate pairs only; reuse the retrieval configuration in inputs. No retrieval is performed by this bundle.'}
        (run_dir / 'bundle.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (run_dir / 'threshold_sweep.json').write_text(json.dumps(sweep, indent=2) + '\n')
        report['runs'][label] = result
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'run': label, 'threshold': threshold, 'dev': result['development'],
                          'paired': result['paired_vs_baseline'], 'pairs_per_second': result['feature_benchmark']['median_pairs_per_second']}), flush=True)
    report['best_development_configuration'] = max(report['runs'], key=lambda k: report['runs'][k]['development']['macro_F0.5'])
    report['selection_caveat'] = 'Historical 3000/1000/1000 sample used for exploration; no 10k-training or untouched-confirmation claim. Confirm on Person 2 fresh split before promotion.'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sample', 'pairs', 'split', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--allowed', type=Path, help='Exact finalized candidate pair TSV; preserves historical K=20 pool.')
    run(p.parse_args())
