import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--external-gpu', choices=['yes','no','unknown'], default='unknown')
args = parser.parse_args()
report = {"platform": platform.platform(), "cpu_cores": os.cpu_count(), "cpu_name": platform.processor(), "disk": shutil.disk_usage(ROOT)._asdict(), "ram_bytes": None, "cuda_available": False, "gpus": [], "external_gpu_available": "unknown", "mode": "C until external GPU access is confirmed", "errors": []}
try:
    import psutil
    report["ram_bytes"] = psutil.virtual_memory().total
except Exception as exc:
    report["errors"].append(f"RAM detection: {exc}")
try:
    import torch
    report["cuda_available"] = torch.cuda.is_available()
    if report["cuda_available"]:
        report["gpus"] = [{"name": torch.cuda.get_device_name(i), "vram_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
        report["mode"] = "A"
    report["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
except ImportError:
    report["errors"].append("PyTorch not installed; no CUDA probe available")
if args.external_gpu == 'yes':
    report['external_gpu_available'] = 'User-confirmed external access; allocation must be checked'
    if not report['cuda_available']:
        report['mode'] = 'B'
elif args.external_gpu == 'no':
    report['external_gpu_available'] = False
files = list((ROOT / "data").rglob("*.tsv"))
report["dataset"] = {"tsv_count": len(files), "total_bytes": sum(p.stat().st_size for p in files)}
report["runtime_estimates"] = {"lexical": "Pending actual record count and sample benchmark", "bge_m3": "Not launched; requires verified model, sample throughput and budget gate", "reranker": "Not launched; requires candidate count, verified model, sample throughput and budget gate", "fine_tuning": "No CPU training planned; external CUDA GPU and successful pretrained validation required"}
(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "artifacts/compute_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
