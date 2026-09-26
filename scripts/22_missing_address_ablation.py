"""Isolated missing-address ablation; never writes the frozen checkpoint."""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import evaluate
from src.features.fuzzy import fuzzy_features
from src.normalize import normalize
from src.predict import predict, select_threshold

spec = importlib.util.spec_from_file_location("sample_experiment", Path(__file__).with_name("12_sample_experiment.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def corrected_features(qn, qa, qc, tn, ta, tc):
    """NaN means no comparison was possible; HGB learns missingness explicitly."""
    return experiment.cheap_features(qn, qa, qc, tn, ta, tc, neutral_missing_address=True) + fuzzy_features(qn, qa, tn, ta, neutral_missing_address=True)


def score_pairs(pairs_path, sample_path, allowed=None):
    sample = {r["entity_id"]: r for r in map(json.loads, sample_path.read_text().splitlines())}
    query = {key: (normalize(row["business_name"]), normalize(row["business_address"]), normalize(row["country"])) for key, row in sample.items()}
    rows = []
    with pairs_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["source1_entity_id"], row["target_entity_id"])
            if allowed is not None and key not in allowed:
                continue
            feature = corrected_features(*query[key[0]], normalize(row["target_name"]), normalize(row["target_address"]), normalize(row["target_country"]))
            rows.append((key[0], key[1], row["target_source"], feature))
    pairs = pd.DataFrame([(s, t, source) for s, t, source, _ in rows], columns=["source1_entity_id", "target_entity_id", "target_source"])
    features = np.asarray([values for _, _, _, values in rows], dtype=np.float32)
    truth = {key: set(row["matches"]) for key, row in sample.items()}
    labels = np.asarray([target in truth[source] for source, target, _, _ in rows], dtype=np.int8)
    return pairs, features, labels, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--missed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = pd.read_csv(args.run / "candidate_pairs_internal.tsv", sep="\t", dtype=str)
    allowed = set(zip(original.source1_entity_id, original.target_entity_id))
    pairs, x, y, truth = score_pairs(args.candidates / "pairs.jsonl", args.candidates / "sample_s1.jsonl", allowed)
    assert len(pairs) == len(original) and set(zip(pairs.source1_entity_id, pairs.target_entity_id)) == allowed
    split = json.loads((args.run / "split.json").read_text())
    train_mask = pairs.source1_entity_id.isin(split["train"]).to_numpy()
    model = HistGradientBoostingClassifier(max_iter=200, max_depth=4, max_leaf_nodes=15, learning_rate=.08, l2_regularization=1, early_stopping=False, random_state=2026)
    model.fit(x[train_mask], y[train_mask])
    pairs["score"] = model.predict_proba(x)[:, 1]
    threshold, sweep = select_threshold(pairs, truth, split["calibration"])
    dev = evaluate(truth, predict(pairs, threshold), split["validation"])
    dev_at_old_threshold = evaluate(truth, predict(pairs, .64), split["validation"])
    with args.missed.open() as stream:
        missed = list(csv.DictReader(stream))
    scores = dict(zip(zip(pairs.source1_entity_id, pairs.target_entity_id), pairs.score))
    moved = sum(scores[(r["source1_entity_id"], r["target_entity_id"])] >= .64 for r in missed)
    cpairs, cx, _, ctruth = score_pairs(args.confirmation / "pairs.jsonl", args.confirmation / "sample_s1.jsonl")
    cpairs["score"] = model.predict_proba(cx)[:, 1]
    confirmation = evaluate(ctruth, predict(cpairs, threshold))
    confirmation_at_old_threshold = evaluate(ctruth, predict(cpairs, .64))
    report = {"feature_schema": "original 16 columns; absent address comparisons = NaN, address_missing retained", "train_pairs": int(train_mask.sum()), "threshold": float(threshold), "dev": dev, "confirmation": confirmation, "dev_at_old_threshold": dev_at_old_threshold, "confirmation_at_old_threshold": confirmation_at_old_threshold, "original_515_above_0.64": moved, "original_missed_count": len(missed), "threshold_sweep": sweep, "cheap_only": True, "uses_fuzzy": True, "neutral_missing_address": True}
    joblib.dump({"model": model, "threshold": float(threshold), "source_run": str(args.run), "uses_fuzzy": True, "neutral_missing_address": True}, args.output / "model.joblib")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "validation.json").write_text(json.dumps({**report, "validation": dev}, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "threshold_sweep"}, indent=2))


if __name__ == "__main__":
    main()
