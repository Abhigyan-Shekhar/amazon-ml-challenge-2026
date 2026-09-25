"""Export/import held-out pairs from the scalable sample experiments, no GPU locally."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluate import evaluate
from src.predict import predict,select_threshold


def export(candidates,run,output):
    output.mkdir(parents=True,exist_ok=False)
    sample={r['entity_id']:r for r in map(json.loads,(candidates/'sample_s1.jsonl').read_text().splitlines())}
    split=json.loads((run/'split.json').read_text());held=set(split['calibration'])|set(split['validation'])
    final=pd.read_csv(run/'candidate_pairs_internal.tsv',sep='\t',dtype=str)
    wanted=set(map(tuple,final[final.source1_entity_id.isin(held)][['source1_entity_id','target_entity_id']].to_numpy()))
    seen=set()
    with (output/'pairs.jsonl').open('w') as out,(candidates/'pairs.jsonl').open() as source:
        for line in source:
            r=json.loads(line);key=(r['source1_entity_id'],r['target_entity_id'])
            if key not in wanted:continue
            if key in seen:raise ValueError('Duplicate pair')
            seen.add(key);q=sample[key[0]]
            out.write(json.dumps({'source1_entity_id':key[0],'target_entity_id':key[1],'text_a':f"RECORD A\n[NAME] {q['business_name']}\n[ADDRESS] {q['business_address']}\n[COUNTRY] {q['country']}",'text_b':f"RECORD B\n[NAME] {r['target_name']}\n[ADDRESS] {r['target_address']}\n[COUNTRY] {r['target_country']}"},ensure_ascii=False)+'\n')
    if seen!=wanted:raise ValueError('Missing final candidate text')
    shutil.copy('scripts/gpu/score_pairs.py',output/'score_pairs.py')
    shutil.copy(run/'split.json',output/'split.json')
    metadata={'baseline_run':str(run.resolve()),'pairs_sha256':hashlib.sha256((output/'pairs.jsonl').read_bytes()).hexdigest(),'pair_count':len(seen),'truth':{i:sample[i]['matches'] for i in held},'countries':{i:sample[i]['country'] for i in held},'baseline_metrics':json.loads((run/'validation.json').read_text())['results']}
    (output/'evaluation_metadata.json').write_text(json.dumps(metadata,indent=2))
    (output/'README.md').write_text('''# GPU validation job — no training yet

Upload this directory privately to Kaggle/Colab in accordance with challenge data rules. Enable a CUDA GPU. Run:

```sh
pip install torch transformers huggingface_hub sentencepiece
python score_pairs.py --pairs pairs.jsonl --output-dir reranker_pretrained --budget-seconds 3600
```

The script verifies live model metadata before weights, saves compliance evidence, benchmarks representative pairs, and refuses full inference if its projection exceeds the budget. Do not reduce the pool after scoring; changing K requires a new export and retrieval validation. Download the complete reranker_pretrained directory and import locally with scripts/14_gpu_sample_job.py import. No model training or leaderboard upload is performed.

Use evaluation_metadata.json and split.json only for local evaluation; labels are not part of model text. The 1,000 calibration S1s and 1,000 validation S1s are disjoint from supervised lexical training.
''')
    print(json.dumps({'output':str(output),'pairs':len(seen)}))


def import_scores(job,cache,output):
    metadata=json.loads((job/'evaluation_metadata.json').read_text());manifest=json.loads((cache/'cache_manifest.json').read_text())
    if metadata['pairs_sha256']!=manifest['pairs_sha256'] or hashlib.sha256((job/'pairs.jsonl').read_bytes()).hexdigest()!=manifest['pairs_sha256']:raise ValueError('Mismatched pair content hash')
    expected={(r['source1_entity_id'],r['target_entity_id']) for r in map(json.loads,(job/'pairs.jsonl').read_text().splitlines())}
    scores=pd.read_json(cache/'scores.jsonl',lines=True)
    actual=set(map(tuple,scores[['source1_entity_id','target_entity_id']].to_numpy()))
    if actual!=expected or len(scores)!=len(expected):raise ValueError('Scores do not exactly cover final candidates')
    if not np.isfinite(scores.score).all() or not scores.score.between(0,1).all():raise ValueError('Invalid scores')
    split=json.loads((job/'split.json').read_text());truth={i:set(ts) for i,ts in metadata['truth'].items()}
    threshold,sweep=select_threshold(scores,truth,split['calibration']);predicted=predict(scores,threshold)
    report={'threshold':threshold,'validation':evaluate(truth,predicted,split['validation']),'slices':{country:evaluate(truth,predicted,[i for i in split['validation'] if metadata['countries'][i]==country]) for country in set(metadata['countries'].values())},'threshold_sweep':sweep,'model_manifest':manifest,'baseline':metadata['baseline_metrics'],'status':'comparison only; requires singleton/slice review and final compliance recheck before promotion'}
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['validation']))

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('export');e.add_argument('--candidates',type=Path,required=True);e.add_argument('--run',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    i=sub.add_parser('import');i.add_argument('--job',type=Path,required=True);i.add_argument('--cache',type=Path,required=True);i.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='export':export(a.candidates,a.run,a.output)
    else:import_scores(a.job,a.cache,a.output)
