"""Export the exact scored candidate pool to portable JSONL; no model download."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, read_sources

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--partition', choices=['train','test'], required=True)
parser.add_argument('--candidates', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
sources = read_sources(load_config(args.config)[args.partition])
lookup = {row.entity_id: row.combined_text for frame in sources.values() for row in frame.itertuples()}
candidates = pd.read_csv(args.candidates, sep='\t', dtype={'source1_entity_id': str, 'target_entity_id': str}, keep_default_na=False)
if candidates[['source1_entity_id','target_entity_id']].duplicated().any():
    raise ValueError('Duplicate pairs')
s1 = set(sources['s1'].entity_id)
targets = set(sources['s2'].entity_id) | set(sources['s3'].entity_id)
if set(candidates.source1_entity_id)-s1 or set(candidates.target_entity_id)-targets:
    raise ValueError('Invalid source or target IDs')
with Path(args.output).open('x') as f:
    for row in candidates.itertuples():
        f.write(json.dumps({'source1_entity_id': row.source1_entity_id, 'target_entity_id': row.target_entity_id, 'text_a': 'RECORD A\n'+lookup[row.source1_entity_id], 'text_b': 'RECORD B\n'+lookup[row.target_entity_id]}, ensure_ascii=False)+'\n')
print(json.dumps({'pairs':len(candidates), 'sha256':hashlib.sha256(Path(args.output).read_bytes()).hexdigest()}))
