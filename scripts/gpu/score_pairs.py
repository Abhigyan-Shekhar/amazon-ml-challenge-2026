"""Standalone Kaggle/Colab CUDA inference; verify, benchmark, gate, cache.
Install: pip install torch transformers huggingface_hub sentencepiece
No external business data is fetched. Only approved pretrained model files.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--model', default='BAAI/bge-reranker-v2-m3')
parser.add_argument('--budget-seconds', type=float, required=True)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--sample-size', type=int, default=128)
parser.add_argument('--max-length', type=int, default=512)
args = parser.parse_args()
if args.budget_seconds <= 0 or args.batch_size < 1 or args.sample_size < 1:
    raise ValueError('Budget, batch size and sample size must be positive')
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=False)
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU required; stop and use the lexical baseline locally')
from huggingface_hub import HfApi, hf_hub_download
# Remote metadata and README are checked BEFORE loading model weights.
info = HfApi().model_info(args.model)
card = info.card_data.to_dict() if info.card_data else {}
license_id = card.get('license')
safe = info.safetensors
parameter_count = getattr(safe, 'total', None)
if license_id not in ('mit', 'apache-2.0') or not isinstance(parameter_count, int) or not 0 < parameter_count <= 8_000_000_000:
    raise RuntimeError(f'Compliance cannot be established: license={license_id}, count={parameter_count}. Stop for repository review.')
readme_path = hf_hub_download(args.model, 'README.md', revision=info.sha)
readme = Path(readme_path).read_text()
(output/'model_repository_README.md').write_text(readme)
evidence = {'model_name':args.model, 'source':f'https://huggingface.co/{args.model}/tree/{info.sha}', 'revision':info.sha, 'license':license_id, 'parameter_count':parameter_count, 'parameter_count_method':'Hugging Face safetensors metadata', 'verified_at':datetime.now(timezone.utc).isoformat(), 'model_card_sha256':hashlib.sha256(readme.encode()).hexdigest(), 'requires_final_packaging_reverification':True}
(output/'model_compliance.md').write_text('# Repository verification before experiment\n\n```json\n'+json.dumps(evidence,indent=2)+'\n```\n\nRepository README saved alongside this file. Repeat verification before final packaging.\n')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
start = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(args.model, revision=info.sha, trust_remote_code=False)
model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=info.sha, trust_remote_code=False).to('cuda').eval()
actual = sum(p.numel() for p in model.parameters())
if actual > 8_000_000_000:
    raise RuntimeError('Loaded parameter count exceeds competition limit')
evidence['loaded_parameter_count'] = actual
pairs_path = Path(args.pairs)
pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
if not pairs:
    raise ValueError('No pairs to score')
keys = [(p['source1_entity_id'],p['target_entity_id']) for p in pairs]
if len(set(keys)) != len(keys):
    raise ValueError('Duplicate candidate pairs')

def score(batch):
    encoded = tokenizer([p['text_a'] for p in batch], [p['text_b'] for p in batch], padding=True, truncation=True, max_length=args.max_length, return_tensors='pt').to('cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        logits = model(**encoded).logits
    if logits.shape[-1] != 1:
        raise ValueError('Expected a scalar reranker logit')
    return logits[:,0].float().sigmoid().cpu().tolist()

# Representative length quantiles, deterministically spread through the pool.
ordered = sorted(range(len(pairs)), key=lambda i: len(pairs[i]['text_a'])+len(pairs[i]['text_b']))
n = min(args.sample_size, len(pairs))
sample = [pairs[ordered[round(i*(len(pairs)-1)/max(1,n-1))]] for i in range(n)]
score(sample[:min(args.batch_size,n)])
torch.cuda.synchronize()
bench_start = time.monotonic()
for offset in range(0,n,args.batch_size):
    score(sample[offset:offset+args.batch_size])
torch.cuda.synchronize()
elapsed = time.monotonic()-bench_start
rate = n/elapsed
projected = len(pairs)/rate*1.5
benchmark = {'sample_pairs':n, 'pairs_per_second':rate, 'projected_full_seconds_with_50pct_margin':projected, 'budget_seconds':args.budget_seconds, 'setup_seconds':time.monotonic()-start, 'gpu':torch.cuda.get_device_name(0), 'max_length':args.max_length}
(output/'benchmark.json').write_text(json.dumps(benchmark,indent=2))
if projected > args.budget_seconds:
    raise RuntimeError('Runtime gate failed. Reduce K or use lexical fallback; full inference was NOT launched.')
full_start = time.monotonic()
with (output/'scores.jsonl.partial').open('x') as f:
    for offset in range(0,len(pairs),args.batch_size):
        if time.monotonic()-full_start >= args.budget_seconds:
            raise RuntimeError('Inference time box reached; partial cache preserved, not eligible for import')
        batch = pairs[offset:offset+args.batch_size]
        for p, value in zip(batch, score(batch)):
            f.write(json.dumps({'source1_entity_id':p['source1_entity_id'], 'target_entity_id':p['target_entity_id'], 'score':value})+'\n')
        f.flush()
(output/'scores.jsonl.partial').rename(output/'scores.jsonl')
evidence.update({'pairs_sha256':hashlib.sha256(pairs_path.read_bytes()).hexdigest(), 'pair_count':len(pairs), 'benchmark':benchmark, 'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','huggingface_hub')}})
(output/'cache_manifest.json').write_text(json.dumps(evidence,indent=2))
print(f'Completed {len(pairs)} pairs; import scores.jsonl and cache_manifest.json together.')
