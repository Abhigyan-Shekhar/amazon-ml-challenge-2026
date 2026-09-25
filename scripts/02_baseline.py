"""Calibration-tuned, S1-held-out char TF-IDF baseline. No neural training."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, read_sources, read_truth
from src.blocking.char_tfidf import retrieve
from src.evaluate import evaluate, blocking_metrics
from src.split import grouped_split
from src.predict import predict, select_threshold
from src.generate_submission import write_submission
from src.normalize import VERSION


def run(config_path, output_root, experiment_id):
    start = time.monotonic()
    config = load_config(config_path)
    run_dir = Path(output_root) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    train = read_sources(config['train'])
    truth = read_truth(config['train']['labels'], train)
    calibration, validation = grouped_split(train['s1'].entity_id, config.get('seed', 2026))
    settings = dict(k=config.get('candidate_k_per_source', 50), batch_size=config.get('batch_size', 64), max_features=config.get('max_features', 300000))
    candidates, timing = retrieve(train, **settings)
    candidates.to_csv(run_dir/'train_candidates.tsv', sep='\t', index=False)
    threshold, sweep = select_threshold(candidates, truth, calibration)
    predicted = predict(candidates, threshold)
    metrics = evaluate(truth, predicted, validation)
    metrics.update(blocking_metrics(truth, candidates.groupby('source1_entity_id').target_entity_id.agg(set).to_dict(), validation))
    countries = train['s1'].set_index('entity_id').normalized_country.to_dict()
    slices = {'countries': {}, 'sources': {}}
    for country in sorted(set(countries.values())):
        ids = [i for i in validation if countries[i] == country]
        slices['countries'][country] = evaluate(truth, predicted, ids)
    for source in ('s2', 's3'):
        target_ids = set(train[source].entity_id)
        slices['sources'][source] = evaluate({i: t & target_ids for i, t in truth.items()}, {i: p & target_ids for i, p in predicted.items()}, validation)
    # Country transfer is a threshold-transfer sanity check for this unsupervised matcher.
    transfer = {}
    for origin, destination in [('india', 'us'), ('us', 'india')]:
        origin_ids = [i for i in calibration if countries[i] == origin]
        destination_ids = [i for i in validation if countries[i] == destination]
        if origin_ids and destination_ids:
            t, _ = select_threshold(candidates, truth, origin_ids)
            transfer[f'{origin}_to_{destination}'] = {'threshold': t, 'metrics': evaluate(truth, predict(candidates, t), destination_ids)}
        else:
            transfer[f'{origin}_to_{destination}'] = {'status': 'not enough exact country-labelled groups; check actual country strings'}
    max_scores = candidates.groupby('source1_entity_id').score.max().to_dict()
    with (run_dir/'singleton_scores.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['source1_entity_id', 'split', 'true_match_count', 'max_candidate_score'])
        writer.writerows((i, 'calibration' if i in set(calibration) else 'validation', len(truth[i]), max_scores.get(i, '')) for i in truth)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    hashes = {partition: {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in config[partition].items()} for partition in ('train', 'test') if partition in config}
    report = {'experiment_id': experiment_id, 'timestamp': datetime.now(timezone.utc).isoformat(), 'git_commit': commit, 'candidate_method': 'char_tfidf', 'candidate_K': settings['k'], 'threshold': threshold, 'metrics': metrics, 'slices': slices, 'country_transfer': transfer, 'retrieval_timing': timing, 'split': {'calibration': calibration, 'validation': validation}, 'threshold_sweep': sweep, 'input_sha256': hashes, 'hardware': platform.platform(), 'official_validator_status': 'unavailable; internal validator only', 'runtime': time.monotonic()-start}
    (run_dir/'validation.json').write_text(json.dumps(report, indent=2))
    # Freeze before test inference; never reuse an existing experiment directory.
    manifest = {'experiment_id': experiment_id, 'git_commit': commit, 'candidate_config': settings, 'normalization_config': VERSION, 'retrieval_config': 'char TF-IDF (3,5), separate S2/S3, no country filter, nonzero overlap only', 'RRF_config': None, 'candidate_K': settings['k'], 'model_name': 'char_tfidf_cosine', 'checkpoint_path': None, 'model_parameter_count': 0, 'model_license': 'no pretrained model; sklearn BSD-3-Clause', 'feature_config': ['name + address'], 'threshold': threshold, 'validation_macro_F0.5': metrics['macro_F0.5'], 'validation_slices': slices, 'random_seeds': [config.get('seed', 2026)], 'package_versions': {x: importlib.metadata.version(x) for x in ('numpy', 'pandas', 'scipy', 'scikit-learn')}, 'input_sha256': hashes, 'submission_format_status': 'PROVISIONAL: official schema unconfirmed'}
    (run_dir/'final_manifest.json').write_text(json.dumps(manifest, indent=2))
    if 'test' in config:
        test = read_sources(config['test'])
        test_candidates, test_timing = retrieve(test, **settings)
        test_candidates.to_csv(run_dir/'test_scored_candidates.tsv', sep='\t', index=False)
        write_submission(run_dir/'outputs', test, test_candidates, predict(test_candidates, threshold))
        (run_dir/'test_runtime.json').write_text(json.dumps(test_timing, indent=2))
    row = {'experiment_id': experiment_id, 'timestamp': report['timestamp'], 'git_commit': commit, 'candidate_method': 'char_tfidf', 'candidate_K': settings['k'], **metrics, 'reranker_model': 'none', 'reranker_checkpoint': '', 'feature_set': 'name_address_char_3_5', 'threshold': threshold, 'India_F0.5': slices['countries'].get('india', {}).get('macro_F0.5'), 'US_F0.5': slices['countries'].get('us', {}).get('macro_F0.5'), 'S2_metric': slices['sources']['s2']['macro_F0.5'], 'S3_metric': slices['sources']['s3']['macro_F0.5'], 'runtime': time.monotonic()-start, 'hardware': report['hardware'], 'notes': 'holdout S1; micro precision/recall; canonical provisional outputs; no official validator'}
    log_path = Path(output_root)/'results.csv'
    exists = log_path.exists()
    with log_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps({'run_dir': str(run_dir), 'validation': metrics, 'threshold': threshold}, indent=2))
    return run_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', default='artifacts/experiments')
    parser.add_argument('--experiment-id', default=datetime.now(timezone.utc).strftime('char_%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    run(args.config, args.output_root, args.experiment_id)
