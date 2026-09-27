"""Measure lexical matching and a grouped logistic feature model on real S1s."""
import argparse,csv,hashlib,json,sys,time,subprocess
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize
from src.evaluate import evaluate,blocking_metrics
from src.predict import predict,select_threshold
from src.split import grouped_split

FEATURES=['combined_cosine','name_cosine','address_cosine','name_jaccard','address_jaccard','name_exact','address_exact','numeric_jaccard','numeric_conflict','country_match','name_length_ratio','address_length_ratio','address_missing','name_address_product']

def cheap_features(qn,qa,qc,tn,ta,tc,neutral_missing_address=False):
    qnt,tnt=set(qn.split()),set(tn.split());qat,tat=set(qa.split()),set(ta.split())
    qnum={t for t in qat if any(c.isdigit() for c in t)};tnum={t for t in tat if any(c.isdigit() for c in t)}
    jac=lambda a,b:len(a&b)/max(1,len(a|b))
    ratio=lambda a,b:min(len(a),len(b))/max(1,len(a),len(b))
    values=[jac(qnt,tnt),jac(qat,tat),float(qn==tn),float(bool(qa) and qa==ta),jac(qnum,tnum),float(bool(qnum and tnum) and not qnum&tnum),float(qc==tc),ratio(qn,tn),ratio(qa,ta),float(not qa or not ta)]
    if neutral_missing_address and (not qa or not ta):
        for index in (1,3,4,5,8):
            values[index]=float('nan')
    return values

def run(directory,output,ranking="char",k=50):
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    sample=[json.loads(l) for l in (directory/'sample_s1.jsonl').read_text().splitlines()]
    qids=[r['entity_id'] for r in sample]; lookup={x:i for i,x in enumerate(qids)};truth={r['entity_id']:set(r['matches']) for r in sample}
    qnames=[normalize(r['business_name']) for r in sample];qaddrs=[normalize(r['business_address']) for r in sample]
    qcountries=[normalize(r['country']) for r in sample]
    pairs=pd.read_json(directory/'pairs.jsonl',lines=True)
    curves=[]
    for floor in (0,.25,.35,.45,.55):
        for sweep_k in (10,20,50,100):
            pool=pairs[(pairs.blocking_score>=floor)&(pairs.blocking_rank<=sweep_k)]
            mapping=pool.groupby('source1_entity_id').target_entity_id.agg(set).to_dict()
            curves.append({'floor':floor,'k':sweep_k,'pairs':len(pool),**blocking_metrics(truth,mapping)})
    (output/'blocking_curves.json').write_text(json.dumps(curves,indent=2))
    print('Blocking curves',json.dumps(curves),flush=True)
    if ranking=='blocking':
        pairs=pairs.sort_values(['source1_entity_id','target_source','blocking_score','target_entity_id'],ascending=[True,True,False,False]).groupby(['source1_entity_id','target_source'],sort=False).head(k).reset_index(drop=True)
    pairs['qi']=pairs.source1_entity_id.map(lookup)
    pairs['name']=pairs.target_name.map(normalize);pairs['address']=pairs.target_address.map(normalize);pairs['country']=pairs.target_country.map(normalize)
    fit_targets=pairs[['target_entity_id','name','address']].drop_duplicates('target_entity_id').sample(n=min(30000,pairs.target_entity_id.nunique()),random_state=2026)
    corpora={'combined':[n+' '+a for n,a in zip(qnames,qaddrs)]+(fit_targets.name+' '+fit_targets.address).tolist(),'name':qnames+fit_targets.name.tolist(),'address':qaddrs+fit_targets.address.tolist()}
    vectorizers={}; cosines=[]
    for field,corpus in corpora.items():
        v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=150000,dtype=np.float32);v.fit(corpus);vectorizers[field]=v
        qtext=[n+' '+a for n,a in zip(qnames,qaddrs)] if field=='combined' else qnames if field=='name' else qaddrs
        qmatrix=v.transform(qtext);scores=np.empty(len(pairs),dtype=np.float32)
        for offset in range(0,len(pairs),5000):
            batch=pairs.iloc[offset:offset+5000];texts=(batch.name+' '+batch.address) if field=='combined' else batch[field]
            scores[offset:offset+len(batch)]=np.asarray(qmatrix[batch.qi.to_numpy()].multiply(v.transform(texts)).sum(axis=1)).ravel()
        cosines.append(scores)
        print(field,'scored',len(pairs),'pairs',time.monotonic()-start,flush=True)
    base=np.empty((len(pairs),10),dtype=np.float32)
    for i,row in enumerate(pairs.itertuples()):
        qi=row.qi;base[i]=cheap_features(qnames[qi],qaddrs[qi],qcountries[qi],row.name,row.address,row.country)
    features=np.column_stack([*cosines,base,cosines[1]*cosines[2]])
    # Freeze final pool after within-block char ranking, independently for each source.
    pairs['score']=cosines[0]
    rank_column='score' if ranking=='char' else 'blocking_score'
    selected=pairs.sort_values(['source1_entity_id','target_source',rank_column,'target_entity_id'],ascending=[True,True,False,ranking=='char']).groupby(['source1_entity_id','target_source'],sort=False).head(k).index.to_numpy()
    pairs=pairs.loc[selected].reset_index(drop=True);features=features[selected]
    train,held=grouped_split(qids,2026,.4);calibration,validation=grouped_split(held,2027,.5)
    split={'train':train,'calibration':calibration,'validation':validation}
    (output/'split.json').write_text(json.dumps(split,indent=2))
    train_mask=pairs.source1_entity_id.isin(train).to_numpy()
    labels=np.array([r.target_entity_id in truth[r.source1_entity_id] for r in pairs.itertuples()],dtype=np.int8)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=1.0,random_state=2026))
    model.fit(features[train_mask],labels[train_mask])
    probability=model.predict_proba(features)[:,1]
    cheap_model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=1.0,random_state=2026))
    cheap_model.fit(features[train_mask,3:13],labels[train_mask])
    cheap_probability=cheap_model.predict_proba(features[:,3:13])[:,1]
    countries={r['entity_id']:r['country'] for r in sample}; results={}
    pools=pairs.groupby('source1_entity_id').target_entity_id.agg(set).to_dict()
    def slices(predicted,ids):
        out={country:evaluate(truth,predicted,[i for i in ids if countries[i]==country]) for country in sorted(set(countries.values()))}
        for source in ('S2-','S3-'):
            out[source]=evaluate({i:{t for t in ts if t.startswith(source)} for i,ts in truth.items()},{i:{t for t in ts if t.startswith(source)} for i,ts in predicted.items()},ids)
        return out
    for method,scores in [('char_cosine',pairs.score.to_numpy()),('logistic_features',probability),('cheap_logistic',cheap_probability)]:
        scored=pairs[['source1_entity_id','target_entity_id','target_source']].copy();scored['score']=scores
        threshold,sweep=select_threshold(scored,truth,calibration)
        predictions=predict(scored,threshold)
        metrics=evaluate(truth,predictions,validation);metrics.update(blocking_metrics(truth,pools,validation))
        results[method]={'threshold':threshold,'validation':metrics,'calibration':evaluate(truth,predictions,calibration),'slices':slices(predictions,validation),'threshold_sweep':sweep}
        scored.to_csv(output/f'{method}_scores.tsv',sep='\t',index=False)
        print(method,json.dumps(results[method]['validation']),flush=True)
    winner=max(results,key=lambda x:results[x]['calibration']['macro_F0.5'])
    # Method choice uses calibration; holdout is for reporting, not threshold fitting.
    bundle={'model':model,'cheap_model':cheap_model,'cheap_threshold':results['cheap_logistic']['threshold'],'vectorizers':vectorizers,'features':FEATURES,'threshold':results['logistic_features']['threshold'],'selected_method':winner,'ranking':ranking,'k_per_source':k,'normalization':'NFKC lowercase punctuation-to-space','sample_ids':qids}
    joblib.dump(bundle,output/'lexical_model.joblib')
    np.savez_compressed(output/'features.npz',features=features,labels=labels)
    pairs[['source1_entity_id','target_entity_id','target_source']].to_csv(output/'candidate_pairs_internal.tsv',sep='\t',index=False)
    report={'sample_s1':len(qids),'sample_method':'uniform reservoir seed 2026; candidates searched against all training targets','split_sizes':{k:len(v) for k,v in split.items()},'final_k_per_source':k,'ranking':ranking,'blocking':json.loads((directory/'benchmark.json').read_text()),'results':results,'selected_by_calibration':winner,'runtime_seconds':time.monotonic()-start,'feature_names':FEATURES,'neural_model':None,'official_validation':'pending full test inference','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (output/'validation.json').write_text(json.dumps(report,indent=2))
    print('Selected',winner,'total seconds',time.monotonic()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,default=Path('artifacts/candidates/rare_token_v1'));p.add_argument('--output',type=Path,default=Path('artifacts/experiments/sample_v1'));p.add_argument('--ranking',choices=['char','blocking'],default='char');p.add_argument('--k',type=int,default=50);a=p.parse_args();run(a.candidates,a.output,a.ranking,a.k)
