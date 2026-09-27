"""Retrain frozen XGBoost controls on completed, lossless reverse unions.

Run 24_feature_ablation.py first. Model/K selection and thresholds use calibration;
all development results are exploratory. No production bundle is modified.
"""
import argparse
import csv
import hashlib
import io
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Reuse the exact frozen loader, calibration, metrics and baseline utilities.
control = importlib.import_module('24_feature_ablation')
from src.features import competition
from src.features.matching import PairFeatureExtractor, FAMILY_NAMES, VERSION
from src.blocking.union import pair_key, unique_records
from src.evaluate import evaluate, blocking_metrics
from src.predict import predict, select_threshold

MODEL_VERSION = 'reverse-union-xgb-v1'
ROOT = Path(__file__).resolve().parents[1]
sha = control.sha


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def input_hashes(args):
    return {k: sha(getattr(args, k)) for k in ('sample', 'pairs', 'split', 'allowed')}


def completed_unions(directory, hashes, forward, queries):
    """Validate provenance and lossless membership without consulting labels."""
    report = json.loads((directory / 'report.json').read_text())
    manifest = report['reverse_manifest']
    if not report.get('complete') or not manifest.get('complete') or manifest.get('benchmark_only'):
        raise ValueError('Completed full-scan unions required')
    if manifest['k'] < 8:
        raise ValueError('Full top-8 context required')
    expected = {'sample': hashes['sample'], 'split': hashes['split'],
                'forward_pairs': hashes['allowed'], 'forward_cache': hashes['pairs']}
    if any(report['inputs'].get(k) != v for k, v in expected.items()):
        raise ValueError('Union input hashes differ from frozen baseline')
    if (manifest['sample_sha256'] != hashes['sample'] or
            manifest['forward_pairs_sha256'] != hashes['allowed']):
        raise ValueError('Reverse manifest differs from frozen inputs')
    frozen = unique_records(forward)
    previous = set(frozen)
    unions = {}
    for k in (1, 3, 8):
        entry = report['unions'][str(k)]
        path = directory / f'union_k{k}.jsonl'
        if entry['path'] != path.name or sha(path) != entry['sha256']:
            raise ValueError('Union file/hash mismatch')
        rows = read_rows(path)
        current = unique_records(rows)
        if not previous <= current.keys():
            raise ValueError('Unions must be nested and preserve every forward edge')
        for key, row in current.items():
            if key[0] not in queries:
                raise ValueError('Unknown S1 in union')
            routes = row['retrieval_routes']
            if ('forward_structured' in routes) != (key in frozen):
                raise ValueError('Forward route membership changed')
            if key in frozen:
                base = frozen[key]
                for field in ('target_source', 'target_name', 'target_address', 'target_country',
                              'blocking_score', 'blocking_rank'):
                    if row.get(field) != base.get(field):
                        raise ValueError('Frozen forward text/score/rank changed: ' + field)
                for alias, original in (('forward_score', 'blocking_score'), ('forward_rank', 'blocking_rank')):
                    if row[alias] != base.get(original):
                        raise ValueError('Forward alias changed')
            elif any(row.get(f) is not None for f in
                     ('forward_score', 'forward_rank', 'blocking_score', 'blocking_rank')):
                raise ValueError('Reverse-only candidates cannot acquire forward metadata')
            if 'reverse_sparse' in routes:
                if (row.get('reverse_rank') is None or not 1 <= row['reverse_rank'] <= k or
                        row.get('reverse_score') is None or
                        row['reverse_score'] <= manifest['score_floor_strict']):
                    raise ValueError('Reverse route violates rank/score contract')
        competition.transform(rows)  # Also reject invalid missingness/rank encodings.
        unions[k] = rows
        previous = set(current)
    return unions, report


def frame_for(rows):
    return pd.DataFrame([{k: r[k] for k in
                         ('source1_entity_id', 'target_entity_id', 'target_source')} for r in rows])


def candidate_sets(rows):
    pools = defaultdict(set)
    for row in rows:
        pools[row['source1_entity_id']].add(row['target_entity_id'])
    return pools


def pool_metrics(truth, pool, ids):
    counts = [len(pool.get(i, set())) for i in ids]
    return {'s1_count': len(ids), 'candidate_count': sum(counts),
            'true_links': sum(len(truth[i]) for i in ids),
            'found_true_links': sum(len(truth[i] & pool.get(i, set())) for i in ids),
            'counts_per_s1': dict(zip(('p50', 'p95', 'p99', 'max'),
                                      np.quantile(counts, [.5, .95, .99, 1]).tolist())) if ids else {},
            **blocking_metrics(truth, pool, ids)}


def diagnostics(rows, truth, predicted, ids, queries, base_pool):
    pool = candidate_sets(rows)
    missing = {r['source1_entity_id'] for r in rows if not r['target_address'].strip()}
    result = control.slices(truth, predicted, ids, queries, missing)
    groups = {f'country:{c}': [i for i in ids if queries[i]['country'] == c]
              for c in sorted({queries[i]['country'] for i in ids})}
    groups.update({f'true_matches:{n}': [i for i in ids if
                   (len(truth[i]) == int(n) if n != '2+' else len(truth[i]) >= 2)]
                   for n in ('0', '1', '2+')})
    groups['forward_missed_any_truth'] = [i for i in ids if not truth[i] <= base_pool.get(i, set())]
    groups['union_added_candidates'] = [i for i in ids if pool.get(i, set()) - base_pool.get(i, set())]
    for name, members in groups.items():
        result[name] = {'s1': len(members), **evaluate(truth, predicted, members),
                        **pool_metrics(truth, pool, members)}
    # Source-level link metrics include truth missing from the candidate pool.
    for source in ('s2', 's3'):
        source_truth = {i: {t for t in values if t.lower().startswith(source + '-')} for i, values in truth.items()}
        source_pred = {i: {t for t in values if t.lower().startswith(source + '-')} for i, values in predicted.items()}
        source_pool = candidate_sets([r for r in rows if r['target_source'] == source])
        result['target_source:' + source] = {**evaluate(source_truth, source_pred, ids),
                                            **pool_metrics(source_truth, source_pool, ids)}
    return result


def feature_matrix(extractor, rows, queries, with_competition):
    if with_competition and any('reverse_rank_censored' not in r or 'retrieval_routes' not in r for r in rows):
        raise ValueError('Competition models require finalized union metadata')
    chunks = [extractor.transform(rows[i:i+4096], queries) for i in range(0, len(rows), 4096)]
    values = np.vstack(chunks) if chunks else extractor.transform([], queries)
    return np.column_stack((values, competition.transform(rows))) if with_competition else values


def load_bundle(directory):
    bundle = json.loads((directory / 'bundle.json').read_text())
    if bundle['feature_version'] != VERSION:
        raise ValueError('Lexical feature version mismatch')
    added = bundle.get('competition_version')
    if added not in (None, competition.VERSION):
        raise ValueError('Competition version mismatch')
    for name in ('model', 'extractor'):
        if sha(directory / (name + '.json')) != bundle[name + '_sha256']:
            raise ValueError('Bundle integrity mismatch')
    extractor = PairFeatureExtractor.load(directory / 'extractor.json')
    names = extractor.feature_names + (competition.FEATURE_NAMES if added else [])
    if names != bundle['feature_names']:
        raise ValueError('Feature schema mismatch')
    model = XGBClassifier()
    model.load_model(directory / 'model.json')
    return bundle, extractor, model


def train_run(directory, rows, queries, split, truth, params, added, hashes, baseline, base_pool):
    start = time.perf_counter()
    directory.mkdir()
    extractor = PairFeatureExtractor(tuple(FAMILY_NAMES)).fit([queries[i] for i in split['train']])
    tick = time.perf_counter()
    values = feature_matrix(extractor, rows, queries, added)
    feature_seconds = time.perf_counter() - tick
    frame = frame_for(rows)
    train = frame.source1_entity_id.isin(split['train']).to_numpy()
    labels = np.array([r['target_entity_id'] in truth[r['source1_entity_id']] for r in rows], dtype=np.int8)
    if len(np.unique(labels[train])) != 2:
        raise ValueError('Training pool must contain both classes')
    model = XGBClassifier(**params)
    tick = time.perf_counter()
    model.fit(values[train], labels[train])
    fit_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    frame['score'] = model.predict_proba(values)[:, 1]
    score_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    threshold, sweep = select_threshold(frame, truth, split['calibration'])
    threshold_seconds = time.perf_counter() - tick
    predicted = predict(frame, threshold)
    extractor.save(directory / 'extractor.json')
    model.save_model(directory / 'model.json')
    bundle = {'model_version': MODEL_VERSION, 'feature_version': VERSION,
              'competition_version': competition.VERSION if added else None,
              'feature_names': extractor.feature_names + (competition.FEATURE_NAMES if added else []),
              'threshold': float(threshold), 'model_params': params, 'production_promoted': False,
              'inputs': hashes, 'train_ids_sha256': extractor.train_ids_sha256,
              'model_sha256': sha(directory / 'model.json'),
              'extractor_sha256': sha(directory / 'extractor.json')}
    write_json(directory / 'bundle.json', bundle)
    write_json(directory / 'threshold_sweep.json', sweep)
    _, restored, loaded = load_bundle(directory)
    probe = feature_matrix(restored, rows[:100], queries, added)
    np.testing.assert_allclose(probe, values[:100], equal_nan=True)
    np.testing.assert_array_equal(loaded.predict_proba(probe)[:, 1], frame.score.to_numpy()[:100])
    frame['selected'] = frame.score >= threshold
    frame.to_csv(directory / 'scores.tsv', sep='\t', index=False)
    pool = candidate_sets(rows)
    result = {'feature_count': values.shape[1], 'threshold': float(threshold),
              'candidate_count': len(rows), 'train_pairs': int(train.sum()),
              'train_positives': int(labels[train].sum()),
              'routes': dict(Counter('+'.join(r.get('retrieval_routes', ['forward_structured'])) for r in rows)),
              'calibration': evaluate(truth, predicted, split['calibration']),
              'development': evaluate(truth, predicted, split['validation']),
              'candidate_metrics': {name: pool_metrics(truth, pool, ids) for name, ids in split.items()},
              'development_slices': diagnostics(rows, truth, predicted, split['validation'], queries, base_pool),
              'paired_vs_forward47': control.paired_interval(truth, baseline, predicted, split['validation']),
              'runtime_seconds': {'feature_transform': feature_seconds, 'fit': fit_seconds,
                                  'predict': score_seconds, 'threshold_search': threshold_seconds,
                                  'total_including_io_validation_reporting': time.perf_counter() - start},
              'model_sha256': bundle['model_sha256']}
    write_json(directory / 'report.json', result)
    return result


def compare(args):
    start = time.perf_counter()
    queries, forward, split = control.load_inputs(args.sample, args.pairs, args.split, args.allowed)
    hashes = input_hashes(args)
    baseline_report = json.loads((args.baseline / 'report.json').read_text())
    if any(baseline_report['inputs'][k]['sha256'] != v for k, v in hashes.items()):
        raise ValueError('Baseline inputs do not match')
    base_dir = args.baseline / 'name+address+views'
    bundle, extractor, model = load_bundle(base_dir)
    if extractor.families != tuple(FAMILY_NAMES):
        raise ValueError('Expected all-family 47-feature baseline')
    if any(bundle['inputs'][k]['sha256'] != v for k, v in hashes.items()):
        raise ValueError('Baseline bundle inputs do not match')
    expected_extractor = PairFeatureExtractor(tuple(FAMILY_NAMES)).fit([queries[i] for i in split['train']])
    if extractor.train_ids_sha256 != expected_extractor.train_ids_sha256:
        raise ValueError('Baseline training IDs differ from frozen split')
    if bundle['model_params'] != baseline_report['model_params']:
        raise ValueError('Baseline model configuration differs from report')
    unions, retrieval_report = completed_unions(args.unions, hashes, forward, queries)
    # Only after validating all dependencies do we create an output directory.
    args.output.mkdir(parents=True, exist_ok=False)
    truth = {i: set(r['matches']) for i, r in queries.items()}
    frame = frame_for(forward)
    frame['score'] = model.predict_proba(feature_matrix(extractor, forward, queries, False))[:, 1]
    baseline = predict(frame, bundle['threshold'])
    for name, ids in (('calibration', split['calibration']), ('development', split['validation'])):
        measured = evaluate(truth, baseline, ids)
        expected = baseline_report['runs']['name+address+views'][name]
        for metric in ('macro_F0.5', 'precision', 'recall'):
            if not np.isclose(measured[metric], expected[metric], rtol=0, atol=1e-12):
                raise ValueError('Saved baseline predictions do not reproduce recorded metrics')
    base_pool = candidate_sets(forward)
    report = {'complete': False, 'status': 'EXPLORATORY_NOT_PRODUCTION',
              'model_version': MODEL_VERSION, 'feature_version': VERSION,
              'competition_version': competition.VERSION, 'inputs': hashes,
              'baseline_report_sha256': sha(args.baseline / 'report.json'),
              'union_report_sha256': sha(args.unions / 'report.json'),
              'union_sha256': {str(k): sha(args.unions / f'union_k{k}.jsonl') for k in unions},
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'implementation_sha256': {p: sha(ROOT / p) for p in ('scripts/28_reverse_model_evaluation.py',
                   'scripts/24_feature_ablation.py', 'src/features/competition.py', 'src/features/matching.py',
                   'src/evaluate.py', 'src/predict.py')},
              'versions': {p: version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'rapidfuzz', 'anyascii')},
              'python': platform.python_version(), 'platform': platform.platform(),
              'split_sizes': {k: len(v) for k, v in split.items()},
              'model_params': baseline_report['model_params'],
              'policy': 'Fit training S1 only; thresholds and model/K selection on calibration only. Development exploratory. No truth injection or production promotion.',
              'runtime_scope': 'Training/evaluation only. Retrieval/union runtime copied separately from producer.',
              'retrieval_manifest': retrieval_report['reverse_manifest'],
              'union_build_seconds': retrieval_report['seconds'],
              'forward47': baseline_report['runs']['name+address+views'], 'runs': {}}
    # One forward-only competition control uses the same complete top-8 context.
    indexed = unique_records(unions[8])
    enriched = [indexed[pair_key(row)] for row in forward]
    configurations = [('forward47+competition', enriched, True)]
    for k, rows in unions.items():
        configurations.extend([(f'union_k{k}_47', rows, False), (f'union_k{k}_47+competition', rows, True)])
    for name, rows, added in configurations:
        result = train_run(args.output / name, rows, queries, split, truth,
                           report['model_params'], added, {**hashes, 'union_report': report['union_report_sha256']},
                           baseline, base_pool)
        report['runs'][name] = result
        write_json(args.output / 'report.json', report)
        print(json.dumps({'run': name, 'threshold': result['threshold'], 'development': result['development']}), flush=True)
    eligible = {'forward47': report['forward47'], **report['runs']}
    report['selected_by_calibration'] = max(eligible, key=lambda k: (eligible[k]['calibration']['macro_F0.5'], k))
    report.update(complete=True, total_seconds=time.perf_counter() - start)
    write_json(args.output / 'report.json', report)


def summarize_forward(args):
    """Re-score saved controls and add pool/slice counts to the unchanged report."""
    if args.output.exists():
        raise FileExistsError(args.output)
    queries, rows, split = control.load_inputs(args.sample, args.pairs, args.split, args.allowed)
    hashes = input_hashes(args)
    report = json.loads((args.baseline / 'report.json').read_text())
    if any(report['inputs'][k]['sha256'] != v for k, v in hashes.items()):
        raise ValueError('Baseline input mismatch')
    truth = {i: set(r['matches']) for i, r in queries.items()}
    pool = candidate_sets(rows)
    report['candidate_metrics'] = {name: pool_metrics(truth, pool, ids) for name, ids in split.items()}
    report['reproduction'] = {}
    for label in ('baseline', 'name+address+views'):
        directory = args.baseline / label
        bundle, extractor, model = load_bundle(directory)
        tick = time.perf_counter()
        values = feature_matrix(extractor, rows, queries, False)
        feature_seconds = time.perf_counter() - tick
        frame = frame_for(rows)
        tick = time.perf_counter()
        frame['score'] = model.predict_proba(values)[:, 1]
        score_seconds = time.perf_counter() - tick
        predicted = predict(frame, bundle['threshold'])
        for name, ids in (('calibration', split['calibration']), ('development', split['validation'])):
            if evaluate(truth, predicted, ids) != report['runs'][label][name]:
                raise ValueError('Saved model does not reproduce report')
        frame.to_csv(directory / 'scores.tsv', sep='\t', index=False)
        report['reproduction'][label] = {
            'model_sha256': bundle['model_sha256'], 'extractor_sha256': bundle['extractor_sha256'],
            'bundle_sha256': sha(directory / 'bundle.json'),
            'feature_transform_seconds': feature_seconds, 'predict_seconds': score_seconds,
            'runtime_scope': 'All frozen candidates; excludes retrieval, fitting, threshold search and I/O.',
            'development_slices': diagnostics(rows, truth, predicted, split['validation'], queries, pool)}
    historical = json.loads((ROOT / 'artifacts/reports/person1_feature_ablation.json').read_text())
    report['matches_historical_inputs'] = report['inputs'] == historical['inputs']
    report['all_eight_historical_metrics_and_thresholds_exact'] = all(
        report['runs'][k][field] == historical['runs'][k][field]
        for k in report['runs'] for field in ('threshold', 'calibration', 'development'))
    report['selected_by_calibration'] = max(report['runs'], key=lambda k: report['runs'][k]['calibration']['macro_F0.5'])
    report['reverse_comparison_status'] = 'PENDING_COMPLETED_UNIONS'
    report['production_promoted'] = False
    report['summary_implementation_sha256'] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)


def validation_subset_hashes(args, split, forward):
    """Reconstruct producer subset bytes from frozen inputs, without union labels.

    Sample/raw-cache lines retain original bytes/order. Cache includes finalized
    validation edges only. The producer TSV uses csv.writer CRLF; the subset split
    uses JSON indent=2 plus newline. Exact expected hashes must match, not merely IDs.
    """
    ids = set(split['validation'])
    keys = {pair_key(r) for r in forward if r['source1_entity_id'] in ids}
    digest = lambda data: hashlib.sha256(data).hexdigest()
    def filtered(path, predicate):
        with path.open('rb') as stream:
            return b''.join(line for line in stream if line.strip() and predicate(json.loads(line)))
    sample = filtered(args.sample, lambda r: r['entity_id'] in ids)
    cache = filtered(args.pairs, lambda r: pair_key(r) in keys)
    out = io.StringIO(newline='')
    with args.allowed.open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames, delimiter='\t', lineterminator='\r\n')
        writer.writeheader()
        writer.writerows(r for r in reader if r['source1_entity_id'] in ids)
    subset_split = (json.dumps({'validation': split['validation']}, indent=2) + '\n').encode()
    return {'sample': digest(sample), 'pairs': digest(cache),
            'allowed': digest(out.getvalue().encode()), 'split': digest(subset_split)}


def validation_only(args):
    """Frozen-model pool diagnostic: no fitting, threshold search or K selection."""
    start = time.perf_counter()
    queries, forward, split = control.load_inputs(args.sample, args.pairs, args.split, args.allowed)
    full_hashes = input_hashes(args)
    baseline_report = json.loads((args.baseline / 'report.json').read_text())
    if any(baseline_report['inputs'][k]['sha256'] != v for k, v in full_hashes.items()):
        raise ValueError('Frozen baseline input hashes differ')
    producer = json.loads((args.unions / 'report.json').read_text())
    manifest = json.loads(args.manifest.read_text())
    if (sha(args.manifest) != producer['inputs']['reverse_manifest'] or
            manifest != producer['reverse_manifest']):
        raise ValueError('Standalone reverse manifest mismatch')
    ids = split['validation']
    if manifest['emitted_s1_count'] != len(ids):
        raise ValueError('Reverse emission scope is not the frozen validation subset')
    subset_hashes = validation_subset_hashes(args, split, forward)
    subset_queries = {i: queries[i] for i in ids}
    subset_forward = [r for r in forward if r['source1_entity_id'] in subset_queries]
    unions, producer = completed_unions(args.unions, subset_hashes, subset_forward, subset_queries)
    truth = {i: set(r['matches']) for i, r in queries.items()}
    base_pool = candidate_sets(subset_forward)
    models = {}
    for label in ('baseline', 'name+address+views'):
        bundle, extractor, model = load_bundle(args.baseline / label)
        if any(bundle['inputs'][k]['sha256'] != v for k, v in full_hashes.items()):
            raise ValueError('Frozen model bundle input hashes differ')
        expected = PairFeatureExtractor(extractor.families).fit([queries[i] for i in split['train']])
        if extractor.train_ids_sha256 != expected.train_ids_sha256:
            raise ValueError('Model training IDs differ from frozen training split')
        if (bundle['model_params'] != baseline_report['model_params'] or
                bundle['threshold'] != baseline_report['runs'][label]['threshold']):
            raise ValueError('Frozen model parameters/threshold differ')
        # Re-score original calibration pool to prove the same fixed policy.
        calibration = [r for r in forward if r['source1_entity_id'] in set(split['calibration'])]
        frame = frame_for(calibration)
        frame['score'] = model.predict_proba(feature_matrix(extractor, calibration, queries, False))[:, 1]
        if evaluate(truth, predict(frame, bundle['threshold']), split['calibration']) != baseline_report['runs'][label]['calibration']:
            raise ValueError('Frozen calibration metrics do not reproduce')
        models[label] = (bundle, extractor, model)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'status': 'EXPLORATORY_VALIDATION_ONLY',
              'scope': {'mode': 'validation-only', 'validation_s1_count': len(ids),
                        'validation_ids_sha256': hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
                        'full_split_comparison': False, 'model_retrained': False,
                        'threshold_reselected': False, 'k_selected': False,
                        'training_and_calibration_pool': 'original frozen forward only',
                        'competition_features_trained': False},
              'purpose': 'Frozen-model candidate-pool diagnostic on the original validation S1s only; not a retrained reverse-feature comparison.',
              'production_promoted': False, 'full_input_hashes': full_hashes,
              'subset_input_hashes': subset_hashes,
              'subset_derivation': 'Original-order validation sample lines; original-order finalized validation cache lines; filtered TSV with csv CRLF; validation-only split JSON indent=2 plus newline. All four must match producer hashes.',
              'producer_report_sha256': sha(args.unions/'report.json'),
              'reverse_manifest_sha256': sha(args.manifest), 'reverse_manifest': manifest,
              'union_sha256': {str(k): sha(args.unions/f'union_k{k}.jsonl') for k in unions},
              'baseline_report_sha256': sha(args.baseline/'report.json'),
              'model_params': baseline_report['model_params'], 'feature_version': VERSION,
              'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              'implementation_sha256': {p: sha(ROOT/p) for p in ('scripts/28_reverse_model_evaluation.py',
                   'src/features/matching.py', 'src/features/competition.py', 'src/evaluate.py', 'src/predict.py')},
              'versions': {p: version(p) for p in ('numpy','pandas','scikit-learn','xgboost','rapidfuzz','anyascii')},
              'python': platform.python_version(), 'platform': platform.platform(),
              'candidate_metrics': {}, 'models': {}, 'runs': {}}
    pools = {'forward': subset_forward, **{f'union_k{k}': rows for k, rows in unions.items()}}
    for pool_name, rows in pools.items():
        pool = candidate_sets(rows)
        report['candidate_metrics'][pool_name] = pool_metrics(truth, pool, ids)
        report['candidate_metrics'][pool_name].update(
            recovered_true_links=sum(len((truth[i] & pool.get(i,set())) - base_pool.get(i,set())) for i in ids),
            lost_forward_edges=sum(len(base_pool.get(i,set()) - pool.get(i,set())) for i in ids))
    for label, (bundle, extractor, model) in models.items():
        report['models'][label] = {'model_sha256': bundle['model_sha256'],
             'extractor_sha256': bundle['extractor_sha256'], 'threshold': bundle['threshold'],
             'feature_names': bundle['feature_names'], 'feature_count': len(bundle['feature_names']),
             'calibration': baseline_report['runs'][label]['calibration'],
             'train_ids_sha256': extractor.train_ids_sha256}
        base_pred = None
        for pool_name, rows in pools.items():
            tick = time.perf_counter()
            values = feature_matrix(extractor, rows, queries, False)
            feature_seconds = time.perf_counter()-tick
            frame = frame_for(rows)
            tick = time.perf_counter()
            frame['score'] = model.predict_proba(values)[:,1]
            prediction_seconds = time.perf_counter()-tick
            predicted = predict(frame, bundle['threshold'])
            metrics = evaluate(truth, predicted, ids)
            if pool_name == 'forward':
                base_pred = predicted
                if metrics != baseline_report['runs'][label]['development']:
                    raise ValueError('Frozen forward validation metrics do not reproduce')
            name = label + '__' + pool_name
            frame['selected'] = frame.score >= bundle['threshold']
            frame.to_csv(args.output/(name+'.tsv'), sep='\t', index=False)
            report['runs'][name] = {'model': label, 'pool': pool_name, 'validation': metrics,
                'candidate_count': len(rows), 'threshold': bundle['threshold'],
                'paired_vs_same_frozen_forward': control.paired_interval(truth, base_pred, predicted, ids),
                'validation_slices': diagnostics(rows, truth, predicted, ids, queries, base_pool),
                'runtime_seconds': {'feature_transform': feature_seconds, 'predict': prediction_seconds},
                'runtime_scope': 'Validation pool only; excludes retrieval, calibration verification and I/O.'}
            print(json.dumps({'run': name, 'validation': metrics}), flush=True)
        # Prove evaluation did not modify either serialized model artifact.
        load_bundle(args.baseline / label)
    report.update(complete=True, total_seconds=time.perf_counter()-start)
    write_json(args.output/'report.json', report)


def score(args):
    bundle, extractor, model = load_bundle(args.bundle)
    queries = {r['entity_id']: r for r in read_rows(args.sample)}
    with args.pairs.open() as stream, args.output.open('x') as out:
        out.write('source1_entity_id\ttarget_entity_id\tscore\tselected\n')
        batch = []
        def emit(rows):
            values = feature_matrix(extractor, rows, queries, bundle.get('competition_version') is not None)
            scores = model.predict_proba(values)[:, 1]
            for row, value in zip(rows, scores):
                out.write(f"{row['source1_entity_id']}\t{row['target_entity_id']}\t{value:.9g}\t{int(value >= bundle['threshold'])}\n")
        for line in stream:
            if line.strip():
                batch.append(json.loads(line))
            if len(batch) == 4096:
                emit(batch)
                batch = []
        if batch:
            emit(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('compare')
    for name in ('sample', 'pairs', 'split', 'allowed', 'baseline', 'unions', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.set_defaults(function=compare)
    p = sub.add_parser('summarize-forward')
    for name in ('sample', 'pairs', 'split', 'allowed', 'baseline', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.set_defaults(function=summarize_forward)
    p = sub.add_parser('validation-only', help='Frozen models and thresholds; validation unions only, no retraining')
    for name in ('sample', 'pairs', 'split', 'allowed', 'baseline', 'unions', 'manifest', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.set_defaults(function=validation_only)
    p = sub.add_parser('score')
    for name in ('bundle', 'sample', 'pairs', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.set_defaults(function=score)
    args = parser.parse_args()
    args.function(args)
