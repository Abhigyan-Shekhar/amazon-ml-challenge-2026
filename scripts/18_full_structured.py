"""Bounded full-test lexical fallback, using the SAME pool/model as its validation.

Exact name-core OR address-token-set blocks; top K per source by token overlap;
only the finalized heap is passed to the frozen matcher and exported.
"""
import argparse,csv,hashlib,heapq,json,math,sys,time,os
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize
from src.blocking.keys import exact_keys,query_keys,target_keys,address_query_keys,address_target_keys
from collections import Counter
# Import a local script explicitly: this model and feature schema are versioned together.
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('sample_experiment',Path(__file__).with_name('12_sample_experiment.py')); experiment=module_from_spec(spec);spec.loader.exec_module(experiment)


def run(root,model_dir,output,budget,memory_gib,method,tree_dir=None,max_key_frequency=100):
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic();process=psutil.Process()
    report=json.loads((model_dir/'validation.json').read_text());bundle=joblib.load(model_dir/'lexical_model.joblib')
    structured=tree_dir is not None
    if report['ranking']!='blocking':raise ValueError('Requires blocking-ranked validation')
    if not structured and not report['blocking'].get('exact_core_address'):raise ValueError('Requires exact-core/address validation')
    if structured and report['blocking'].get('max_key_frequency')!=max_key_frequency:raise ValueError('Validation key-frequency cap mismatch')
    if method != 'cheap_logistic': raise ValueError('This runtime currently supports the benchmarked cheap logistic fallback only')
    k=bundle['k_per_source'];threshold=report['results'][method]['threshold'];pipeline=bundle['cheap_model']
    scaler=pipeline.named_steps['standardscaler'];lr=pipeline.named_steps['logisticregression'];weights=lr.coef_[0]/scaler.scale_;bias=float(lr.intercept_[0]-np.dot(weights,scaler.mean_))
    manifest={'experiment_id':model_dir.name,'model_name':method,'checkpoint_path':str((model_dir/'lexical_model.joblib').resolve()),'checkpoint_sha256':hashlib.sha256((model_dir/'lexical_model.joblib').read_bytes()).hexdigest(),'model_parameter_count':int(lr.coef_.size+lr.intercept_.size),'model_license':'no pretrained model; scikit-learn BSD-3-Clause','candidate_config':{'method':'exact name core OR address token set','k_per_source':k,'ranking':'max(0.65*name_jaccard+0.30*address_jaccard+0.05*name_exact,0.85*address_jaccard+0.15*name_jaccard)','tie_break':'descending target entity ID'},'normalization_config':'NFKC lowercase punctuation spaces; legal suffix removal only in blocking key','retrieval_config':'exact-core/address inverted S1 index; no country filter','RRF_config':None,'feature_config':experiment.FEATURES[3:13],'threshold':threshold,'validation_macro_F0.5':report['results'][method]['validation']['macro_F0.5'],'validation_slices':report['results'][method]['slices'],'random_seeds':[2026,2027],'git_commit':report['git_commit'],'neural_compliance':'not applicable: no pretrained neural model used','selected_as':'runtime-feasible full-test fallback, not necessarily best sample model','test_inputs':{p.name:{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in root.glob('test_source*.tsv')}}
    if structured:
        tree=joblib.load(tree_dir/'model.joblib');tree_report=json.loads((tree_dir/'validation.json').read_text())
        if not tree_report.get('cheap_only'):raise ValueError('Full streaming tree requires token-only model')
        threshold=tree['threshold'];tree_model=tree['model'];uses_fuzzy=bool(tree.get('uses_fuzzy',False));neutral_missing_address=bool(tree.get('neutral_missing_address',False))
        if uses_fuzzy:
            from src.features.fuzzy import fuzzy_features,NAMES as FUZZY_NAMES
        manifest.update({'model_name':'structured_cheap_tree','checkpoint_path':str((tree_dir/'model.joblib').resolve()),'checkpoint_sha256':hashlib.sha256((tree_dir/'model.joblib').read_bytes()).hexdigest(),'threshold':threshold,'validation_macro_F0.5':tree_report['validation']['macro_F0.5'],'validation_slices':tree_report.get('slices',{}),'model_parameter_count':1+sum(len(t.nodes) for iteration in tree_model._predictors for t in iteration),'parameter_count_method':'numeric split thresholds and leaf values plus base logit; not a neural model','candidate_config':{'method':'structured name/address keys, capped by full S1 key frequency','k_per_source':k,'max_key_frequency':max_key_frequency,'ranking':'max name/address token score'},'selected_as':'validated structured token-feature tree','retrieval_config':'bounded hashed query-key index; no country filter','neutral_missing_address':neutral_missing_address})
    if structured and uses_fuzzy:
        manifest['feature_config']=experiment.FEATURES[3:13]+FUZZY_NAMES
        manifest['uses_fuzzy']=True
    manifest['hash_seed']=os.environ.get('PYTHONHASHSEED','process-random')
    manifest['token_frequency_policy']='fit name/address document frequency on full S1 partition without labels'
    (output/'final_manifest.json').write_text(json.dumps(manifest,indent=2))
    query=[];index={}
    name_df=Counter();addr_df=Counter()
    if structured:
        with (root/'test_source1.tsv').open(newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                name_df.update(set(normalize(row['business_name']).split()));addr_df.update(set(normalize(row['business_address']).split()))
        print('Built partition token frequencies',flush=True)
    with (root/'test_source1.tsv').open(newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            n=normalize(row['business_name']);a=normalize(row['business_address']);c=normalize(row['country']);i=len(query)
            query.append((row['entity_id'],n,a,c))
            keys=query_keys(n,a,name_df)|address_query_keys(a,addr_df) if structured else exact_keys(n,a)
            for raw_key in keys:
                key=hash(raw_key) if structured else raw_key
                if key not in index:index[key]=i
                elif isinstance(index[key],int):index[key]=[index[key],i]
                elif index[key] is not None:
                    index[key].append(i)
                    if structured and len(index[key])>max_key_frequency:index[key]=None
            if i%100000==0 and process.memory_info().rss/2**30>memory_gib:raise RuntimeError('Index memory gate exceeded')
        if structured:index={k:v for k,v in index.items() if v is not None}
    # Store compact target IDs/text only for capped candidates. Raw inputs stay untouched.
    heaps=[[[] for _ in query] for _ in (2,3)]
    print(json.dumps({'s1':len(query),'index_keys':len(index),'rss_gib':process.memory_info().rss/2**30,'seconds':time.monotonic()-start}),flush=True)
    scanned=compared=0;retrieval_start=time.monotonic();peak_rss=0
    for si,source in enumerate((2,3)):
        with (root/f'test_source{source}.tsv').open(newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                scanned+=1;n=normalize(row['business_name']);a=normalize(row['business_address'])
                qids=set()
                keys=target_keys(n,a)|address_target_keys(a) if structured else exact_keys(n,a)
                for raw_key in keys:
                    value=index.get(hash(raw_key) if structured else raw_key,())
                    if isinstance(value,int):qids.add(value)
                    else:qids.update(value)
                if qids:
                    nt=set(n.split());at=set(a.split());record=(row['entity_id'],n,a,normalize(row['country']))
                    for qi in qids:
                        _,qn,qa,_=query[qi];qnt=set(qn.split());qat=set(qa.split())
                        nj=len(qnt&nt)/max(1,len(qnt|nt));aj=len(qat&at)/max(1,len(qat|at))
                        score=max(.65*nj+.30*aj+.05*(qn==n),.85*aj+.15*nj)
                        # Same descending score/ID ranking as sample retrieval.
                        item=(score,record);h=heaps[si][qi]
                        if len(h)<k:heapq.heappush(h,item)
                        elif item>h[0]:heapq.heapreplace(h,item)
                        compared+=1
                if scanned%100000==0:
                    rss=process.memory_info().rss/2**30;peak_rss=max(peak_rss,rss)
                    elapsed=time.monotonic()-retrieval_start;projected=elapsed/scanned*9_969_589
                    state={'scanned':scanned,'compared':compared,'rss_gib':rss,'seconds':elapsed,'projected_retrieval_seconds':projected}
                    print(json.dumps(state),flush=True);(output/'progress.json').write_text(json.dumps(state,indent=2))
                    if rss>memory_gib:raise RuntimeError('Memory gate exceeded; no incomplete submission is eligible')
                    if time.monotonic()-start>budget or (scanned==100000 and projected>budget):raise RuntimeError('Runtime gate exceeded; no incomplete submission is eligible')
    del index
    pair_count=matches=0
    if structured:
        # Batch tree inference across S1s to avoid millions of tiny model calls.
        mf=(output/'matching_results.tsv.partial').open('w');cf=(output/'candidate_pairs.tsv.partial').open('w')
        mf.write('source1_entity_id\tmatched_entity_ids\n');cf.write('source1_entity_id\tcandidate_entity_ids\n')
        pending=[];feature_rows=[]
        def flush():
            nonlocal pair_count,matches
            probs=tree_model.predict_proba(np.asarray(feature_rows,dtype=np.float32))[:,1] if feature_rows else []
            offset=0
            for eid,tids in pending:
                chosen=[t for t,p in zip(tids,probs[offset:offset+len(tids)]) if p>=threshold];offset+=len(tids)
                cf.write(eid+'\t'+','.join(sorted(tids))+'\n');mf.write(eid+'\t'+','.join(sorted(chosen))+'\n');pair_count+=len(tids);matches+=len(chosen)
            pending.clear();feature_rows.clear()
        for qi,(eid,qn,qa,qc) in enumerate(query):
            records=[r for si in (0,1) for _,r in sorted(heaps[si][qi],reverse=True)]
            tids=[r[0] for r in records]
            if len(tids)!=len(set(tids)):raise ValueError('Duplicate target')
            feature_rows.extend(experiment.cheap_features(qn,qa,qc,r[1],r[2],r[3],neutral_missing_address=neutral_missing_address)+(fuzzy_features(qn,qa,r[1],r[2],neutral_missing_address=neutral_missing_address) if uses_fuzzy else []) for r in records);pending.append((eid,tids));heaps[0][qi].clear();heaps[1][qi].clear()
            if len(feature_rows)>=20000 or len(pending)>=5000:flush()
            if qi%100000==0:
                print(json.dumps({'scored_s1':qi,'pairs':pair_count,'matches':matches,'total_seconds':time.monotonic()-start}),flush=True)
                if time.monotonic()-start>budget:raise RuntimeError('Time box exceeded; partial outputs not eligible')
        flush();mf.close();cf.close()
        for name in ['matching_results.tsv','candidate_pairs.tsv']:(output/(name+'.partial')).rename(output/name)
        result={'s1_rows':len(query),'candidates':pair_count,'matches':matches,'seconds':time.monotonic()-start,'peak_retrieval_rss_gib':peak_rss,'official_validation':'pending'}
        (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    matching=output/'matching_results.tsv.partial';candidate=output/'candidate_pairs.tsv.partial'
    with matching.open('w') as mf,candidate.open('w') as cf:
        mf.write('source1_entity_id\tmatched_entity_ids\n');cf.write('source1_entity_id\tcandidate_entity_ids\n')
        for qi,(eid,qn,qa,qc) in enumerate(query):
            records=[r for si in (0,1) for _,r in sorted(heaps[si][qi],reverse=True)]
            if records:
                features=np.asarray([experiment.cheap_features(qn,qa,qc,r[1],r[2],r[3]) for r in records],dtype=np.float64)
                logits=features@weights+bias;probability=1/(1+np.exp(-np.clip(logits,-700,700)))
                chosen=[r[0] for r,p in zip(records,probability) if p>=threshold]
            else:chosen=[]
            candidate_ids=[r[0] for r in records]
            if len(candidate_ids)!=len(set(candidate_ids)) or not set(chosen)<=set(candidate_ids):raise ValueError('Candidate invariant failed')
            cf.write(eid+'\t'+','.join(sorted(candidate_ids))+'\n');mf.write(eid+'\t'+','.join(sorted(chosen))+'\n')
            pair_count+=len(records);matches+=len(chosen);heaps[0][qi].clear();heaps[1][qi].clear()
            if qi%100000==0:
                print(json.dumps({'scored_s1':qi,'pairs':pair_count,'matches':matches,'total_seconds':time.monotonic()-start}),flush=True)
                if time.monotonic()-start>budget: raise RuntimeError('Scoring time box exceeded; partial output retained but not eligible for packaging')
    matching.rename(output/'matching_results.tsv');candidate.rename(output/'candidate_pairs.tsv')
    result={'s1_rows':len(query),'candidates':pair_count,'matches':matches,'seconds':time.monotonic()-start,'peak_retrieval_rss_gib':peak_rss,'official_validation':'must run supplied validator next'}
    (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test-dir',type=Path,default=Path('/Users/abhigyanshekhar/Downloads/test'));p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget-seconds',type=float,default=1800);p.add_argument('--memory-gib',type=float,default=10);p.add_argument('--method',default='cheap_logistic');p.add_argument('--tree-dir',type=Path);p.add_argument('--max-key-frequency',type=int,default=100);a=p.parse_args();run(a.test_dir,a.model_dir,a.output,a.budget_seconds,a.memory_gib,a.method,a.tree_dir,a.max_key_frequency)
