"""Publish compact measured reports; keep record-level data and models ignored."""
import csv,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluate import entity_f05,blocking_metrics
from src.predict import predict

out=Path('artifacts/reports');out.mkdir(exist_ok=True)
eda=json.loads(Path('artifacts/validation/full_audit/eda.json').read_text());(out/'dataset_audit.json').write_text(json.dumps(eda,indent=2))
scale=json.loads(Path('artifacts/validation/char_scale_benchmark.json').read_text());(out/'char_scale_benchmark.json').write_text(json.dumps(scale,indent=2))
rows=[];summaries={}
for name in ['sample_v1','exact_core_v1','address_v2','capped100_k20','capped10_k20']:
    path=Path('artifacts/experiments')/name;r=json.loads((path/'validation.json').read_text());r.setdefault('ranking','char')
    summaries[name]={k:r[k] for k in ['sample_s1','split_sizes','final_k_per_source','ranking','blocking','selected_by_calibration','runtime_seconds']};summaries[name]['results']={k:{f:v[f] for f in ['threshold','validation','calibration','slices']} for k,v in r['results'].items()}
    for method,v in r['results'].items():
        rows.append({'experiment_id':name+'/'+method,'timestamp':datetime.fromtimestamp((path/'validation.json').stat().st_mtime,timezone.utc).isoformat(),'git_commit':r['git_commit'],'candidate_method':name,'candidate_K':r['final_k_per_source'],'candidate_pair_recall':v['validation']['candidate_pair_recall'],'entity_complete_coverage':v['validation']['entity_complete_coverage'],'reranker_model':'none','reranker_checkpoint':'','feature_set':method,'threshold':v['threshold'],**{k:v['validation'][k] for k in ['precision','recall','macro_F0.5','singleton_F0.5','matched_F0.5']},'India_F0.5':v['slices']['India']['macro_F0.5'],'US_F0.5':v['slices']['US']['macro_F0.5'],'S2_metric':v['slices']['S2-']['macro_F0.5'],'S3_metric':v['slices']['S3-']['macro_F0.5'],'runtime':r['runtime_seconds'],'hardware':'10-core ARM CPU; 16 GiB RAM','notes':f"5,000 sampled S1 searched against FULL training S2/S3; train/calibration/validation=3000/1000/1000; effective K={r['final_k_per_source']}/source"})
r=json.loads(Path('artifacts/experiments/address_tree_v1/validation.json').read_text());summaries['address_tree_v1']={k:v for k,v in r.items() if k!='threshold_sweep'}
tree={**next(row for row in rows if row['experiment_id']=='address_v2/logistic_features'),'experiment_id':'address_tree_v1','timestamp':datetime.fromtimestamp(Path('artifacts/experiments/address_tree_v1/validation.json').stat().st_mtime,timezone.utc).isoformat(),'candidate_method':'address_v2','feature_set':'14 lexical features, shallow HistGradientBoosting','threshold':r['threshold'],**{k:r['validation'][k] for k in ['precision','recall','macro_F0.5','singleton_F0.5','matched_F0.5']},'India_F0.5':r['slices']['India']['macro_F0.5'],'US_F0.5':r['slices']['US']['macro_F0.5'],'S2_metric':r['slices']['S2-']['macro_F0.5'],'S3_metric':r['slices']['S3-']['macro_F0.5'],'runtime':r['runtime_seconds']};rows.append(tree)
for name,base_name in [('capped100_k20_tree','capped100_k20'),('capped10_k20_tree','capped10_k20'),('capped10_fuzzy_tree','capped10_k20')]:
    tree_path=Path('artifacts/experiments')/name/'validation.json'
    tr=json.loads(tree_path.read_text());summaries[name]={k:v for k,v in tr.items() if k!='threshold_sweep'}
    parent=next(row for row in rows if row['experiment_id']==base_name+'/cheap_logistic')
    rows.append({**parent,'experiment_id':name,'timestamp':datetime.fromtimestamp(tree_path.stat().st_mtime,timezone.utc).isoformat(),'feature_set':('10 token + 6 RapidFuzz features' if tr.get('uses_fuzzy') else '10 token features')+', shallow HistGradientBoosting','threshold':tr['threshold'],**{k:tr['validation'][k] for k in ['precision','recall','macro_F0.5','singleton_F0.5','matched_F0.5']},'India_F0.5':tr['slices']['India']['macro_F0.5'],'US_F0.5':tr['slices']['US']['macro_F0.5'],'S2_metric':tr['slices']['S2-']['macro_F0.5'],'S3_metric':tr['slices']['S3-']['macro_F0.5'],'runtime':tr['runtime_seconds']})
for path in [Path('artifacts/experiments/results.csv'),out/'results.csv']:
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
sample={x['entity_id']:x for x in map(json.loads,Path('artifacts/candidates/structured_address_v2/sample_s1.jsonl').read_text().splitlines())}
truth={i:set(x['matches']) for i,x in sample.items()};split=json.loads(Path('artifacts/experiments/address_v2/split.json').read_text());scored=pd.read_csv('artifacts/experiments/address_tree_v1/scores.tsv',sep='\t');pred=predict(scored,r['threshold']);ids=split['validation']
entity_scores=np.array([entity_f05(truth[i],pred.get(i,set())) for i in ids]);rng=np.random.default_rng(2026);boot=np.array([rng.choice(entity_scores,len(entity_scores),replace=True).mean() for _ in range(1000)])
max_scores=scored.groupby('source1_entity_id').score.max().to_dict()
analysis={'model':'address_tree_v1 (earlier full-character feature model)','validation_s1':len(ids),'singleton_count':sum(not truth[i] for i in ids),'macro_F0.5_bootstrap_95pct_interval':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_seed':2026,'caveat':'sampling uncertainty only; not selection bias or France/domain-shift uncertainty','max_score_quantiles':{group:dict(zip(['min','p25','p50','p75','p95','max'],np.quantile([max_scores.get(i,0) for i in ids if bool(truth[i])==matched],[0,.25,.5,.75,.95,1]).tolist())) for group,matched in [('singleton',False),('matched',True)]}}
(out/'singleton_analysis.json').write_text(json.dumps(analysis,indent=2));(out/'experiment_summary.json').write_text(json.dumps(summaries,indent=2))
print(json.dumps({'earlier_full_character_tree':r['validation'],'uncertainty':analysis},indent=2))
