"""Provisional canonical TSVs. Official serialization must be confirmed."""
import json
from pathlib import Path
import pandas as pd

def validate_submission(sources, candidates, matching):
    s1 = set(sources["s1"].entity_id)
    targets = set(sources["s2"].entity_id) | set(sources["s3"].entity_id)
    if candidates[["source1_entity_id", "target_entity_id"]].duplicated().any():
        raise ValueError("Duplicate candidate pair")
    if set(candidates.source1_entity_id) - s1 or set(candidates.target_entity_id) - targets:
        raise ValueError("Unknown candidate IDs")
    if matching.source1_entity_id.duplicated().any() or set(matching.source1_entity_id) != s1:
        raise ValueError("Output must cover every S1 exactly once")
    pools = candidates.groupby("source1_entity_id").target_entity_id.agg(set).to_dict()
    for row in matching.itertuples():
        values = json.loads(row.matched_entity_ids) if row.matched_entity_ids else []
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("Matches must be a JSON string list")
        if len(values) != len(set(values)) or not set(values) <= pools.get(row.source1_entity_id, set()):
            raise ValueError("Duplicate match or match absent from final candidates")

def write_submission(directory, sources, candidates, predicted):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if set(predicted) - set(sources["s1"].entity_id):
        raise ValueError("Unknown predicted S1")
    pairs = candidates[["source1_entity_id", "target_entity_id"]].copy()
    matching = pd.DataFrame([(i, json.dumps(sorted(predicted.get(i, [])), ensure_ascii=False) if predicted.get(i) else "") for i in sources["s1"].entity_id], columns=["source1_entity_id", "matched_entity_ids"])
    validate_submission(sources, pairs, matching)
    pairs.to_csv(directory / "candidate_pairs.tsv", sep="\t", index=False)
    matching.to_csv(directory / "matching_results.tsv", sep="\t", index=False)
    # Round-trip catches quoting, tab separation and empty-field errors.
    read = lambda name: pd.read_csv(directory/name, sep="\t", dtype=str, keep_default_na=False)
    validate_submission(sources, read("candidate_pairs.tsv"), read("matching_results.tsv"))
