"""Evaluate cached scores on the baseline's identical candidate pool and split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, read_sources, read_truth
from src.evaluate import evaluate
from src.predict import predict, select_threshold

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--baseline-run', required=True)
parser.add_argument('--pairs', required=True)
parser.add_argument('--cache-dir', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
run = Path(args.baseline_run)
cache = Path(args.cache_dir)
manifest = json.loads((cache/'cache_manifest.json').read_text())
if hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest() != manifest['pairs_sha256']:
    raise ValueError('GPU cache does not match exported pair texts')
rows = [json.loads(line) for line in (cache/'scores.jsonl').read_text().splitlines() if line.strip()]
scored = pd.DataFrame(rows)
original = pd.read_csv(run/'train_candidates.tsv', sep='\t', dtype={'source1_entity_id':str,'target_entity_id':str}, keep_default_na=False)
key_cols = ['source1_entity_id','target_entity_id']
keys = lambda df: set(map(tuple,df[key_cols].to_numpy()))
if scored.duplicated(key_cols).any() or keys(scored) != keys(original):
    raise ValueError('Cached scores must cover exactly the final baseline candidates')
if not np.isfinite(scored.score).all() or not scored.score.between(0,1).all():
    raise ValueError('Invalid sigmoid scores')
report = json.loads((run/'validation.json').read_text())
config = load_config(args.config)
for key, path in config['train'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != report['input_sha256']['train'][key]:
        raise ValueError('Training input changed since baseline')
truth = read_truth(config['train']['labels'], read_sources(config['train']))
threshold, sweep = select_threshold(scored, truth, report['split']['calibration'])
metrics = evaluate(truth, predict(scored,threshold), report['split']['validation'])
result = {'threshold':threshold, 'metrics':metrics, 'baseline_metrics':report['metrics'], 'macro_F0.5_delta':metrics['macro_F0.5']-report['metrics']['macro_F0.5'], 'threshold_sweep':sweep, 'cache_manifest':manifest, 'status':'comparison only; no automatic promotion or submission'}
with Path(args.output).open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(metrics,indent=2))
