import json
from pathlib import Path
import pandas as pd
from .normalize import prepare

FIELDS = ["entity_id", "business_name", "business_address", "country"]

def parse_id_list(value):
    """Official comma-separated IDs; accept legacy JSON lists for old fixtures."""
    if not value:
        return []
    values = json.loads(value) if value.startswith('[') else value.split(',')
    if not isinstance(values, list) or any(not isinstance(v, str) or not v or v != v.strip() for v in values):
        raise ValueError('Invalid ID list')
    return values


def read_records(path):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not set(FIELDS) <= set(frame):
        raise ValueError(f"{path}: required columns {FIELDS}; found {list(frame)}")
    if frame.entity_id.eq("").any() or frame.entity_id.duplicated().any():
        raise ValueError(f"{path}: blank or duplicate entity_id")
    return prepare(frame)

def read_sources(paths):
    frames = {source: read_records(paths[source]) for source in ("s1", "s2", "s3")}
    all_ids = [i for f in frames.values() for i in f.entity_id]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Cross-source ID collision: official ID disambiguation is required")
    return frames

def read_truth(path, sources):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not {"source1_entity_id", "matched_entity_ids"} <= set(frame):
        raise ValueError("Labels require source1_entity_id and matched_entity_ids (comma-separated IDs or blank)")
    if frame.source1_entity_id.duplicated().any():
        raise ValueError("Duplicate S1 label rows")
    truth = {}
    allowed = set(sources["s2"].entity_id) | set(sources["s3"].entity_id)
    for row in frame.itertuples():
        values = parse_id_list(row.matched_entity_ids)
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            raise ValueError("matched_entity_ids must contain string IDs")
        if len(values) != len(set(values)) or set(values) - allowed:
            raise ValueError("Duplicate or invalid ground-truth target IDs")
        truth[row.source1_entity_id] = set(values)
    if set(truth) != set(sources["s1"].entity_id):
        raise ValueError("Labels must explicitly cover every S1, including singleton rows; missing labels are not assumed empty")
    return truth

def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for partition in ("train", "test"):
        if partition in config:
            config[partition] = {k: str((path.parent / v).resolve()) for k, v in config[partition].items()}
    return config
