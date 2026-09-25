"""Evaluate a frozen token-feature tree on disjoint S1s; no calibration or fitting."""
import argparse,hashlib,json,sys
from pathlib import Path
import importlib.util,joblib,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize
from src.evaluate import evaluate,blocking_metrics,entity_f05
from src.predict import predict
spec=importlib.util.spec_from_file_location('sample_experiment',Path(__file__).with_name('12_sample_experiment.py'));exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--original-split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sample={r['entity_id']:r for r in map(json.loads,(a.candidates/'sample_s1.jsonl').read_text().splitlines())}
prior=json.loads(a.original_split.read_text());assert not set(sample)&{i for ids in prior.values() for i in ids}
truth={i:set(r['matches']) for i,r in sample.items()};query={i:(normalize(r['business_name']),normalize(r['business_address']),normalize(r['country'])) for i,r in sample.items()}
pairs=pd.read_json(a.candidates/'pairs.jsonl',lines=True);features=np.asarray([exp.cheap_features(*query[r.source1_entity_id],normalize(r.target_name),normalize(r.target_address),normalize(r.target_country)) for r in pairs.itertuples()],dtype=np.float32)
frozen=joblib.load(a.model)
if frozen.get('uses_fuzzy'):
 from src.features.fuzzy import fuzzy_features
 extra=np.asarray([fuzzy_features(query[r.source1_entity_id][0],query[r.source1_entity_id][1],normalize(r.target_name),normalize(r.target_address)) for r in pairs.itertuples()],dtype=np.float32)
 features=np.column_stack([features,extra])
pairs['score']=frozen['model'].predict_proba(features)[:,1];pred=predict(pairs,frozen['threshold']);metrics=evaluate(truth,pred)
pools=pairs.groupby('source1_entity_id').target_entity_id.agg(set).to_dict();metrics.update(blocking_metrics(truth,pools))
slices={country:evaluate(truth,pred,[i for i in sample if sample[i]['country']==country]) for country in sorted({r['country'] for r in sample.values()})}
for source in ['S2-','S3-']:slices[source]=evaluate({i:{t for t in ts if t.startswith(source)} for i,ts in truth.items()},{i:{t for t in ts if t.startswith(source)} for i,ts in pred.items()})
values=np.asarray([entity_f05(truth[i],pred.get(i,set())) for i in sample]);rng=np.random.default_rng(2028);boot=[rng.choice(values,len(values),replace=True).mean() for _ in range(1000)]
report={'status':'fresh disjoint confirmation, no fitting or threshold tuning','s1_count':len(sample),'singleton_count':sum(not ts for ts in truth.values()),'model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'threshold':frozen['threshold'],'metrics':metrics,'slices':slices,'bootstrap_95pct_interval':np.quantile(boot,[.025,.975]).tolist(),'retrieval':json.loads((a.candidates/'benchmark.json').read_text()),'seed':2028}
with a.output.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
