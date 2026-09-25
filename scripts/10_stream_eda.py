"""Audit the entire official dataset without retaining record text in memory."""
import argparse, csv, hashlib, json, math, random, time
from collections import Counter
from pathlib import Path


def distribution(hist):
    n=sum(hist.values())
    if not n: return {'count':0}
    def quantile(q):
        acc=0
        for k,v in sorted(hist.items()):
            acc+=v
            if acc>=max(1,math.ceil(n*q)): return k
    return {'count':n,'mean':sum(k*v for k,v in hist.items())/n,'min':min(hist),'p25':quantile(.25),'p50':quantile(.5),'p75':quantile(.75),'p95':quantile(.95),'max':max(hist)}


def run(root, output, sample_size):
    start=time.monotonic(); output.mkdir(parents=True,exist_ok=True)
    report={'partitions':{},'seed':2026}; train_ids={}; train_countries={}; sampled={}
    for partition in ['train','test']:
        report['partitions'][partition]={}
        for source in ['source1','source2','source3']:
            path=root/partition/f'{partition}_{source}.tsv'
            ids=set(); countries=Counter(); missing=Counter(); names=Counter(); addresses=Counter(); duplicate=0; malformed=0; rows=0
            rng=random.Random(2026); reservoir=[]
            with path.open(newline='',encoding='utf-8') as f:
                reader=csv.DictReader(f,delimiter='\t'); schema=reader.fieldnames
                if schema != ['entity_id','business_name','business_address','country']: raise ValueError((path,schema))
                for row in reader:
                    rows+=1
                    if None in row or any(v is None for v in row.values()): malformed+=1; continue
                    eid=row['entity_id']; duplicate+=eid in ids; ids.add(eid)
                    countries[row['country']]+=1
                    for k,v in row.items():
                        if not v.strip(): missing[k]+=1
                    names[len(row['business_name'])]+=1; addresses[len(row['business_address'])]+=1
                    if source=='source1' and partition=='train':
                        train_countries[eid]=row['country']
                        if len(reservoir)<sample_size: reservoir.append(row)
                        else:
                            j=rng.randrange(rows)
                            if j<sample_size: reservoir[j]=row
            entry={'rows':rows,'schema':schema,'missing':dict(missing),'duplicate_ids':duplicate,'malformed_rows':malformed,'countries':dict(countries),'name_length':distribution(names),'address_length':distribution(addresses),'bytes':path.stat().st_size}
            report['partitions'][partition][source]=entry
            print(json.dumps({'partition':partition,'source':source,**entry}),flush=True)
            if partition=='train': train_ids[source]=ids
            if source=='source1' and partition=='train': sampled={r['entity_id']:r for r in reservoir}
            (output/'eda_partial.json').write_text(json.dumps(report,indent=2))
    hist=Counter(); source_counts=Counter(); reverse=Counter(); seen=set(); duplicate_labels=0; bad_targets=0; duplicate_matches=0; labels={}; country_hist={}
    with (root/'train/train_ground_truth.tsv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f,delimiter='\t')
        if reader.fieldnames!=['source1_entity_id','matched_entity_ids']: raise ValueError('Unexpected truth schema')
        for row in reader:
            eid=row['source1_entity_id']; duplicate_labels+=eid in seen; seen.add(eid)
            targets=row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
            hist[len(targets)]+=1; duplicate_matches+=len(targets)!=len(set(targets))
            ch=country_hist.setdefault(train_countries.get(eid,'UNKNOWN'),Counter()); ch[len(targets)]+=1
            for target in targets:
                source='source2' if target.startswith('S2-') else 'source3' if target.startswith('S3-') else 'invalid'
                source_counts[source]+=1
                if target not in train_ids.get(source,set()): bad_targets+=1
                reverse[target]+=1
            if eid in sampled: labels[eid]=targets
    missing=train_ids['source1']-seen; unknown=seen-train_ids['source1']
    report['labels']={'rows':sum(hist.values()),'match_count_distribution':dict(sorted(hist.items())),'zero':hist[0],'one':hist[1],'many':sum(v for k,v in hist.items() if k>1),'country_match_histograms':{k:dict(v) for k,v in country_hist.items()},'source_link_counts':dict(source_counts),'duplicate_s1_rows':duplicate_labels,'missing_s1_count':len(missing),'unknown_s1_count':len(unknown),'invalid_target_count':bad_targets,'duplicate_match_rows':duplicate_matches,'shared_target_count':sum(v>1 for v in reverse.values()),'max_s1_per_target':max(reverse.values(),default=0),'shared_target_examples':[k for k,v in reverse.items() if v>1][:10]}
    report['runtime_seconds']=time.monotonic()-start
    report['sample']={'method':'uniform reservoir of training S1, seed 2026','size':len(sampled),'labels_complete':len(sampled)==len(labels)}
    (output/'eda.json').write_text(json.dumps(report,indent=2))
    with (output/'sample_s1.jsonl').open('w') as f:
        for eid,row in sampled.items(): f.write(json.dumps({**row,'matches':labels.get(eid)},ensure_ascii=False)+'\n')
    print(json.dumps(report['labels']),flush=True)
    print('Complete in',report['runtime_seconds'],flush=True)
    if missing or unknown or bad_targets or duplicate_labels or duplicate_matches: raise ValueError('Ground-truth audit failed; inspect eda.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('/Users/abhigyanshekhar/Downloads')); p.add_argument('--output',type=Path,default=Path('artifacts/validation/full_audit')); p.add_argument('--sample-size',type=int,default=5000); a=p.parse_args(); run(a.root,a.output,a.sample_size)
