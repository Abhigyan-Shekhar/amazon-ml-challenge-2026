"""Label-independent rare-name-token blocking against the FULL target corpus.

This is a scale fallback before char TF-IDF ranking, not global TF-IDF retrieval.
Sample labels are retained separately and never consulted for candidate selection.
"""
import argparse,csv,heapq,json,sys,time,hashlib
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize
from src.blocking.keys import query_keys,target_keys,address_query_keys,address_target_keys,exact_keys


def tokens(name): return set(normalize(name).split())

def run(root,sample_path,output,cap,address_path=False,exact=False,max_key_frequency=None):
    output.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    sample=[json.loads(x) for x in sample_path.read_text().splitlines()]
    df_path=Path('artifacts/candidates/rare_token_v1/token_df.json')
    with (root/'train_source1.tsv').open('rb') as source_file: source_hash=hashlib.file_digest(source_file,'sha256').hexdigest()
    df_manifest=df_path.with_name('token_df_manifest.json')
    if df_path.exists() and df_manifest.exists() and json.loads(df_manifest.read_text()).get('source_sha256')==source_hash: df=Counter(json.loads(df_path.read_text()))
    else:
        df=Counter()
        with (root/'train_source1.tsv').open(newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'): df.update(tokens(row['business_name']))
    (output/'token_df.json').write_text(json.dumps(df,ensure_ascii=False))
    (output/'token_df_manifest.json').write_text(json.dumps({'source_sha256':source_hash}))
    adf=Counter()
    if address_path:
        with (root/'train_source1.tsv').open(newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'): adf.update(tokens(row['business_address']))
        (output/'address_df.json').write_text(json.dumps(adf,ensure_ascii=False))
    anchors=defaultdict(list); queries=[]
    for i,row in enumerate(sample):
        name=normalize(row['business_name']); addr=normalize(row['business_address']); nt=set(name.split()); at=set(addr.split())
        for key in (exact_keys(name,addr) if exact else query_keys(name,addr,df) | (address_query_keys(addr,adf) if address_path else set())): anchors[key].append(i)
        queries.append((name,addr,nt,at))
    key_counts=Counter()
    if max_key_frequency is not None:
        with (root/'train_source1.tsv').open(newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                name=normalize(row['business_name']);addr=normalize(row['business_address'])
                keys=query_keys(name,addr,df) | (address_query_keys(addr,adf) if address_path else set())
                key_counts.update(k for k in keys if k in anchors)
        removed=sum(v>max_key_frequency for v in key_counts.values())
        anchors={k:v for k,v in anchors.items() if key_counts[k]<=max_key_frequency}
        print(json.dumps({'remaining_keys':len(anchors),'removed_keys':removed,'max_key_frequency':max_key_frequency}),flush=True)
    heaps=[[[] for _ in sample] for _ in (2,3)]; scanned=0; hits=0
    for sidx,source in enumerate((2,3)):
        with (root/f'train_source{source}.tsv').open(newline='') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                scanned+=1; name=normalize(row['business_name']); nt=set(name.split())
                addr=normalize(row['business_address']); at=set(addr.split())
                qids=set()
                for key in (exact_keys(name,addr) if exact else target_keys(name,addr) | (address_target_keys(addr) if address_path else set())): qids.update(anchors.get(key,()))
                if not qids: continue
                record=(row['entity_id'],row['business_name'],row['business_address'],row['country'])
                for i in qids:
                    qn,qa,qnt,qat=queries[i]
                    nj=len(qnt&nt)/max(1,len(qnt|nt)); aj=len(qat&at)/max(1,len(qat|at))
                    score=0.65*nj+0.30*aj+0.05*(qn==name)
                    if address_path or exact: score=max(score,0.85*aj+0.15*nj)
                    item=(score,record)
                    h=heaps[sidx][i]
                    if len(h)<cap: heapq.heappush(h,item)
                    elif item>h[0]: heapq.heapreplace(h,item)
                    hits+=1
                if scanned%500000==0: print(json.dumps({'scanned':scanned,'scored_pairs':hits,'seconds':time.monotonic()-start}),flush=True)
        print(json.dumps({'source_complete':source,'scanned':scanned,'scored_pairs':hits,'seconds':time.monotonic()-start}),flush=True)
    with (output/'pairs.jsonl').open('w') as f:
        for i,row in enumerate(sample):
            for sidx in (0,1):
                for rank,(score,r) in enumerate(sorted(heaps[sidx][i],reverse=True),1):
                    f.write(json.dumps({'source1_entity_id':row['entity_id'],'target_entity_id':r[0],'target_name':r[1],'target_address':r[2],'target_country':r[3],'target_source':f's{sidx+2}','blocking_score':score,'blocking_rank':rank},ensure_ascii=False)+'\n')
    (output/'sample_s1.jsonl').write_text(sample_path.read_text())
    report={'method':('exact sorted name core OR exact sorted address tokens' if exact else 'structured name and address blocks' if address_path else 'structured name blocks'),'k_per_source':cap,'scanned_targets':scanned,'scored_pairs':hits,'sample_s1':len(sample),'seconds':time.monotonic()-start,'label_independent':True,'address_path':address_path,'exact_core_address':exact,'max_key_frequency':max_key_frequency}
    (output/'benchmark.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/Users/abhigyanshekhar/Downloads/train'));p.add_argument('--sample',type=Path,default=Path('artifacts/validation/full_audit/sample_s1.jsonl'));p.add_argument('--output',type=Path,default=Path('artifacts/candidates/structured_v1'));p.add_argument('--cap',type=int,default=100);p.add_argument('--address-path',action='store_true');p.add_argument('--exact',action='store_true');p.add_argument('--max-key-frequency',type=int);a=p.parse_args();run(a.root,a.sample,a.output,a.cap,a.address_path,a.exact,a.max_key_frequency)
