"""Official comma-separated ID lists, one row per Source-1 entity."""
import json
from pathlib import Path
import pandas as pd
from .data import parse_id_list

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
        values = parse_id_list(row.matched_entity_ids)
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
    matching = pd.DataFrame([(i, ",".join(sorted(predicted.get(i, []))) if predicted.get(i) else "") for i in sources["s1"].entity_id], columns=["source1_entity_id", "matched_entity_ids"])
    validate_submission(sources, pairs, matching)
    pools = pairs.groupby("source1_entity_id").target_entity_id.agg(list).to_dict()
    official_pairs = pd.DataFrame([(i, ','.join(sorted(pools.get(i, [])))) for i in sources['s1'].entity_id], columns=['source1_entity_id', 'candidate_entity_ids'])
    official_pairs.to_csv(directory / "candidate_pairs.tsv", sep="\t", index=False)
    matching.to_csv(directory / "matching_results.tsv", sep="\t", index=False)
    # Round-trip catches quoting, tab separation and empty-field errors.
    read = lambda name: pd.read_csv(directory/name, sep="\t", dtype=str, keep_default_na=False)
    roundtrip = read('candidate_pairs.tsv')
    expanded = pd.DataFrame([(r.source1_entity_id, t) for r in roundtrip.itertuples() for t in parse_id_list(r.candidate_entity_ids)], columns=['source1_entity_id','target_entity_id'])
    if roundtrip.source1_entity_id.tolist() != sources['s1'].entity_id.tolist():
        raise ValueError('Candidate output must cover all S1 entities exactly once')
    validate_submission(sources, expanded, read("matching_results.tsv"))
