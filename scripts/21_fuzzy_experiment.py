"""One measured edit-feature addition to the runtime-feasible CPU tree."""
import argparse,json,time,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.features.fuzzy import fuzzy_features,NAMES
from src.normalize import normalize
from src.evaluate import evaluate
from src.predict import predict,select_threshold
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--baseline-tree',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
sample={r['entity_id']:r for r in map(json.loads,(a.candidates/'sample_s1.jsonl').read_text().splitlines())};truth={i:set(r['matches']) for i,r in sample.items()}
query={i:(normalize(r['business_name']),normalize(r['business_address'])) for i,r in sample.items()}
pairs=pd.read_csv(a.run/'candidate_pairs_internal.tsv',sep='\t',dtype=str);indices={(r.source1_entity_id,r.target_entity_id):i for i,r in enumerate(pairs.itertuples())};extra=np.empty((len(pairs),6),dtype=np.float32);seen=set()
with (a.candidates/'pairs.jsonl').open() as f:
 for line in f:
  r=json.loads(line);key=(r['source1_entity_id'],r['target_entity_id'])
  if key not in indices:continue
  extra[indices[key]]=fuzzy_features(*query[key[0]],normalize(r['target_name']),normalize(r['target_address']));seen.add(key)
assert seen==set(indices)
arrays=np.load(a.run/'features.npz');x=np.column_stack([arrays['features'][:,3:13],extra]);y=arrays['labels'];split=json.loads((a.run/'split.json').read_text());mask=pairs.source1_entity_id.isin(split['train']).to_numpy()
feature_seconds=time.monotonic()-start
model=HistGradientBoostingClassifier(max_iter=200,max_depth=4,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=2026);model.fit(x[mask],y[mask]);pairs['score']=model.predict_proba(x)[:,1]
t,sweep=select_threshold(pairs,truth,split['calibration']);pred=predict(pairs,t);cal=evaluate(truth,pred,split['calibration']);val=evaluate(truth,pred,split['validation'])
slices={c:evaluate(truth,pred,[i for i in split['validation'] if sample[i]['country']==c]) for c in ['India','US']}
for source in ['S2-','S3-']:slices[source]=evaluate({i:{v for v in ts if v.startswith(source)} for i,ts in truth.items()},{i:{v for v in ts if v.startswith(source)} for i,ts in pred.items()},split['validation'])
base=json.loads((a.baseline_tree/'validation.json').read_text());retained=cal['macro_F0.5']>base['calibration']['macro_F0.5'] and cal['singleton_F0.5']>=base['calibration']['singleton_F0.5']-.02
report={'model':'shallow HistGradientBoosting + 6 RapidFuzz edit features','base_run':str(a.run),'threshold':t,'calibration':cal,'validation':val,'slices':slices,'retained_by_calibration':retained,'runtime_seconds':time.monotonic()-start,'feature_seconds':feature_seconds,'feature_pairs':len(pairs),'pairs_per_feature_second':len(pairs)/feature_seconds,'fuzzy_features':NAMES,'threshold_sweep':sweep,'cheap_only':True,'uses_fuzzy':True}
joblib.dump({'model':model,'threshold':t,'source_run':str(a.run),'uses_fuzzy':True},a.output/'model.joblib');pairs.to_csv(a.output/'scores.tsv',sep='\t',index=False);(a.output/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
