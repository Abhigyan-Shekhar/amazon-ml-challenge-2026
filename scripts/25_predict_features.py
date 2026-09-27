"""Batch scoring with a Person 1 model bundle; no threshold retuning or fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from xgboost import XGBClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.matching import PairFeatureExtractor, VERSION


def score_batches(bundle_dir, sample_path, pairs_path, output_path, batch_size=4096):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    meta = json.loads((bundle_dir / 'bundle.json').read_text())
    for filename, key in [('model.json', 'model_sha256'), ('extractor.json', 'extractor_sha256')]:
        actual = hashlib.sha256((bundle_dir / filename).read_bytes()).hexdigest()
        if actual != meta[key]:
            raise ValueError(f'Bundle integrity mismatch: {filename}')
    extractor = PairFeatureExtractor.load(bundle_dir / 'extractor.json')
    if meta['feature_version'] != VERSION or meta['feature_names'] != extractor.feature_names:
        raise ValueError('Model/extractor feature schema mismatch')
    model = XGBClassifier()
    model.load_model(bundle_dir / 'model.json')
    if model.n_features_in_ != len(extractor.feature_names):
        raise ValueError('Model feature count mismatch')
    queries = {}
    for line in sample_path.read_text().splitlines():
        row = json.loads(line)
        if row['entity_id'] in queries:
            raise ValueError('Duplicate S1 record')
        queries[row['entity_id']] = row
    with output_path.open('x', newline='') as output, pairs_path.open() as stream:
        writer = csv.writer(output, delimiter='\t')
        writer.writerow(['source1_entity_id', 'target_entity_id', 'score', 'selected'])
        def flush(batch):
            scores = model.predict_proba(extractor.transform(batch, queries))[:, 1]
            for row, score in zip(batch, scores):
                writer.writerow([row['source1_entity_id'], row['target_entity_id'], float(score), int(score >= meta['threshold'])])
        batch = []
        for line in stream:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'sample', 'pairs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4096)
    a = parser.parse_args()
    score_batches(a.bundle, a.sample, a.pairs, a.output, a.batch_size)
