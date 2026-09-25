import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, read_sources, read_truth
from src.split import grouped_split

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = load_config(args.config)
report = {}
for partition in ("train", "test"):
    if partition not in config:
        continue
    sources = read_sources(config[partition])
    report[partition] = {}
    for name, frame in sources.items():
        report[partition][name] = {"rows": len(frame), "schema": list(frame.columns), "blank_fields": {col: int(frame[col].str.strip().eq("").sum()) for col in ("entity_id", "business_name", "business_address", "country")}, "duplicate_ids": int(frame.entity_id.duplicated().sum()), "countries": frame.country.value_counts().to_dict(), "name_length": frame.business_name.str.len().describe().to_dict(), "address_length": frame.business_address.str.len().describe().to_dict()}
    if partition == "train":
        truth = read_truth(config[partition]["labels"], sources)
        reverse = defaultdict(set)
        for i, targets in truth.items():
            for target in targets:
                reverse[target].add(i)
        counts = Counter(map(len, truth.values()))
        report["labels"] = {"match_count_distribution": dict(counts), "zero": counts[0], "one": counts[1], "many": sum(v for k, v in counts.items() if k > 1), "shared_target_count": sum(len(v)>1 for v in reverse.values()), "shared_target_examples": {k: sorted(v) for k,v in list((item for item in reverse.items() if len(item[1])>1))[:20]}, "source_link_counts": {source: sum(len(reverse[i]) for i in sources[source].entity_id) for source in ("s2", "s3")}}
        train, val = grouped_split(sources["s1"].entity_id, config.get("seed", 2026))
        report["split"] = {"seed": config.get("seed", 2026), "calibration_ids": train, "validation_ids": val, "note": "Threshold tuning uses calibration only; held-out validation is reported separately"}
output = Path(config.get("eda_output", "artifacts/validation/eda.json"))
output.parent.mkdir(parents=True, exist_ok=True)
def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k,v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False))
print(f"Saved {output}")
